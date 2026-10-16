import base64
import io
import json
import mimetypes
from pathlib import Path
from typing import Optional, Tuple, Union

import openai
from PIL import Image

from ai_feedback.helpers.model_options_helpers import (
    cast_to_type,
    openai_chat_option_schema,
)
from ai_feedback.helpers.pdf_locate import MARKUS_COORDINATE_SCALE

from .OpenAIRemoteModel import GatewayError, OpenAIRemoteModel, _failure_reason

#: An image as PNG bytes already in memory, or a path to an image file.
ImageInput = Union[bytes, str, Path]


class OpenAIRemotePDFAnnotationModel(OpenAIRemoteModel):
    """An OpenAIRemoteModel that annotates one PDF page at a time.

    Intended for vision models (e.g. Qwen-VL) served through the gateway. Each request
    carries the image of a single page and, optionally, the text of the whole submission,
    so the model can annotate the page with the rest of the document as context. The
    caller renders the page and extracts the text beforehand (see ``helpers.pdf_pages``).

    The model is asked for ``x1, y1, x2, y2`` in pixels of the page image; replies are
    converted to page points and then to MarkUs's PDF annotation coordinates (fractions of
    the page width and height times MARKUS_COORDINATE_SCALE) before they are returned.
    """

    def __init__(
        self,
        remote_url: str = "http://localhost:8000/v1",
        model_name: str = "qwen-38-9b",
    ) -> None:
        """Initialize a client pointed at the gateway.

        Args:
            remote_url: Base URL of the gateway's OpenAI-compatible API
                (the ``/v1`` root).
            model_name: The vision model to request. Must be one of the models
                the gateway is configured to allow.
        """
        super().__init__(remote_url=remote_url, model_name=model_name)

    def generate_response(
        self,
        prompt: str,
        system_instructions: str,
        page_image: ImageInput,
        page_size: Tuple[float, float],
        submission_text: Optional[str] = None,
        page_label: Optional[str] = None,
        model_options: Optional[dict] = None,
        json_schema: Optional[str] = None,
        **kwargs,
    ) -> Tuple[str, str]:
        """Annotate one page, optionally with the whole submission's text as context.

        The user message is sent as: the prompt, the submission text, the page label,
        the page size, then the page image.

        Args:
            prompt (str): The rendered user prompt.
            system_instructions (str): Instructions for the model.
            page_image (ImageInput): The page to annotate, as PNG bytes or an image file path.
            page_size (Tuple[float, float]): The page's (width, height) in PDF points, e.g. from
                ``helpers.pdf_pages.pdf_page_size``. The model's pixel coordinates are converted
                to points on this page, then to MarkUs coordinates.
            submission_text (Optional[str]): Text of the whole submission, for context.
            page_label (Optional[str]): Text placed just before the image, e.g. ``"Page 3 of 10"``.
            model_options (Optional[dict]): Generation options; max_tokens defaults to DEFAULT_MAX_TOKENS.
            json_schema (Optional[str]): Optional path to a response_format schema file.
            **kwargs: Accepted for interface compatibility with the other models; ignored.

        Returns:
            Tuple[str, str]: The prompt and the model's response, with any ``x1, y1, x2, y2``
                converted to MarkUs's 0 to MARKUS_COORDINATE_SCALE page coordinates.
        """
        schema = None
        if json_schema:
            schema_path = Path(json_schema)
            if not schema_path.exists():
                raise FileNotFoundError(f"JSON schema file not found: {schema_path}")
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)

        content = [{"type": "text", "text": prompt}]
        if submission_text:
            content.append({"type": "text", "text": f"<submission_text>\n{submission_text}\n</submission_text>"})
        if page_label:
            content.append({"type": "text", "text": page_label})
        mime, image_bytes = self._image_bytes(page_image)
        image_width, image_height = Image.open(io.BytesIO(image_bytes)).size
        content.append(
            {
                "type": "text",
                "text": f"The page image is {image_width} x {image_height} pixels. "
                "Give x1, y1, x2, y2 in pixels of this image, origin top-left.",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"},
            }
        )

        response = self._call_openai_with_content(content, system_instructions, model_options, schema)
        page_width, page_height = page_size
        points_per_pixel = (page_width / image_width, page_height / image_height)
        return prompt, self._scale_to_markus(response, points_per_pixel, page_size)

    @staticmethod
    def _scale_to_markus(
        response: str, points_per_pixel: Tuple[float, float], page_size: Tuple[float, float]
    ) -> str:
        """Convert annotation boxes in ``response`` from image pixels to MarkUs coordinates.

        Each annotation with numeric x1, y1, x2, y2 is reordered so (x1, y1) is the top-left
        corner, converted from pixels to page points, then to a fraction of the page times
        MARKUS_COORDINATE_SCALE, and clamped to the page. A reply that is not JSON, and
        annotations without coordinates, are left as-is.
        """
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return response
        annotations = data.get("annotations") if isinstance(data, dict) else data
        if not isinstance(annotations, list):
            return response

        (x_points, y_points), (width, height) = points_per_pixel, page_size

        def scale(pixels: float, points_per_pixel: float, size: float) -> int:
            fraction = pixels * points_per_pixel / size
            return min(max(round(fraction * MARKUS_COORDINATE_SCALE), 0), MARKUS_COORDINATE_SCALE)

        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            box = [annotation.get(key) for key in ("x1", "y1", "x2", "y2")]
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box):
                continue
            x1, y1, x2, y2 = box
            annotation["x1"], annotation["x2"] = (scale(x, x_points, width) for x in (min(x1, x2), max(x1, x2)))
            annotation["y1"], annotation["y2"] = (scale(y, y_points, height) for y in (min(y1, y2), max(y1, y2)))
        return json.dumps(data)

    @staticmethod
    def _image_bytes(image: ImageInput) -> Tuple[str, bytes]:
        """The MIME type and bytes of ``image``; bytes are taken to be PNG, files are typed by extension."""
        if isinstance(image, bytes):
            return "image/png", image
        path = Path(image)
        return mimetypes.guess_type(path.name)[0] or "image/png", path.read_bytes()

    def _call_openai_with_content(
        self, content: list[dict], system_instructions: str, model_options: Optional[dict], schema: Optional[dict]
    ) -> str:
        """Like OpenAIRemoteModel._call_openai, but with a multi-part user message."""
        model_options = cast_to_type(openai_chat_option_schema, dict(model_options or {}))
        model_options.setdefault("max_tokens", self.DEFAULT_MAX_TOKENS)
        response_format = {"type": "json_schema", "json_schema": schema} if schema else None
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": content},
                ],
                response_format=response_format,
                **model_options,
            )
        except openai.APIError as exc:
            raise GatewayError(_failure_reason(exc)) from exc
        return response.choices[0].message.content
