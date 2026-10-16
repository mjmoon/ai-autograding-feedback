import json
from pathlib import Path
from typing import Optional, Tuple

from .helpers.pdf_locate import place_annotations
from .helpers.pdf_pages import (
    extract_pdf_submission_text,
    pdf_page_count,
    pdf_page_size,
    render_pdf_page,
)
from .helpers.template_utils import render_prompt_template
from .models.Model import Model
from .models.OpenAIRemotePDFAnnotationModel import OpenAIRemotePDFAnnotationModel


def process_pdf_page(
    model: Model, args, prompt: str, system_instructions: str, marking_instructions: Optional[str] = None
) -> Tuple[str, str]:
    """
    Annotates one page of a PDF submission, with the whole submission's text as context.

    Renders page ``args.page`` of ``args.submission`` as an image and sends it with the
    text of every page. If the prompt already includes the submission through
    ``{file_contents}``, the text is not sent a second time. The model's comments are then
    placed on the page (see ``helpers.pdf_locate``) and returned as MarkUs PDF annotations.

    Args:
        model (Model): Must be an OpenAIRemotePDFAnnotationModel (``--provider openai-remote-pdf``).
        args: Command-line argument namespace; uses submission, page, solution, test_output,
            json_schema and model_options.
        prompt (str): The initial user prompt.
        system_instructions (str): instructions for the model
        marking_instructions (str, optional): Marking instructions to replace {marking_instructions} placeholder

    Returns:
        Tuple[str, str]: The request, and ``{"annotations": [...]}`` as JSON, ready for MarkUs's
            ``upload_annotations``; the model's raw response if it could not be parsed.

    Raises:
        FileNotFoundError: If the submission file does not exist.
        SystemExit: If the provider, submission type or page number is unsuitable.
    """
    if not isinstance(model, OpenAIRemotePDFAnnotationModel):
        raise SystemExit("The pdf_page scope requires --provider openai-remote-pdf.")
    submission_file = Path(args.submission)
    if not submission_file.is_file():
        raise FileNotFoundError(f"Submission file '{submission_file}' not found.")
    if submission_file.suffix.lower() != ".pdf":
        raise SystemExit(f"The pdf_page scope requires a .pdf submission, got '{submission_file.name}'.")
    if args.page is None:
        raise SystemExit("The pdf_page scope requires --page.")

    page_count = pdf_page_count(submission_file)
    try:
        page_image = render_pdf_page(submission_file, args.page)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    solution_file = Path(args.solution) if args.solution else None
    test_output = Path(args.test_output) if args.test_output else None
    rendered_prompt = render_prompt_template(
        prompt,
        submission=submission_file,
        solution=solution_file,
        test_output=test_output,
        marking_instructions=marking_instructions,
    )
    submission_text = None if "{file_contents}" in prompt else extract_pdf_submission_text(submission_file)

    request, response = model.generate_response(
        prompt=rendered_prompt,
        system_instructions=system_instructions,
        page_image=page_image,
        page_size=pdf_page_size(submission_file, args.page),
        submission_text=submission_text,
        page_label=f"Annotate this page: page {args.page} of {page_count}.",
        json_schema=args.json_schema,
        model_options=args.model_options,
    )
    placed = place_annotations(submission_file, args.page, response)
    if placed is None:
        return request, response
    annotations, _placements = placed
    return request, json.dumps({"annotations": annotations})
