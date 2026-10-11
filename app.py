
"""
app.py
-------
Gradio web app for the AI IT Helpdesk project.

Loads already-trained models from helpdesk_core.HelpdeskModels.
Run train_models.py before starting this application.
"""

import gradio as gr
from helpdesk_core import HelpdeskModels


# --------------------------------------------------
# Load trained models
# --------------------------------------------------

try:
    engine = HelpdeskModels()
    LOAD_ERROR = None
except FileNotFoundError as e:
    engine = None
    LOAD_ERROR = str(e)


# --------------------------------------------------
# Format historical solution recommendations
# --------------------------------------------------

def format_recommendations(recommendations):
    if recommendations is None or recommendations.empty:
        return (
            "## No matching historical solutions found\n\n"
            "Try describing the problem in more detail."
        )

    cards = [
        "## Recommended Solutions from Past Tickets",
        (
            "These recommendations are based on similar historical tickets. "
            "They are not guaranteed fixes for the current issue."
        ),
    ]

    for i, (_, row) in enumerate(
        recommendations.head(3).iterrows(), start=1
    ):
        ticket_id = str(row.get("number", "N/A"))
        description = str(
            row.get("short_description", "No description available")
        )
        close_notes = row.get("close_notes", "")
        similarity = row.get("similarity", None)

        if close_notes is None or str(close_notes).strip().lower() in (
            "", "nan", "none"
        ):
            resolution = "No recorded resolution is available."
        else:
            close_notes = str(close_notes).replace("\\n", "\n")
            lines = [
                line.strip().lstrip("-• ").strip()
                for line in close_notes.splitlines()
                if line.strip()
            ]
            resolution = "\n".join(f"- {line}" for line in lines)

        try:
            similarity_text = f"{float(similarity):.1%}"
        except (ValueError, TypeError):
            similarity_text = "Not available"

        cards.append(
            f"""
---

### {i}. {description}

**Historical Ticket:** `{ticket_id}`

**Text Similarity:** {similarity_text}

**Recorded Resolution**

{resolution}
"""
        )

    return "\n\n".join(cards)


# --------------------------------------------------
# Analyze ticket
# --------------------------------------------------

def analyze_ticket(short_description, ticket_type, software_system):
    if engine is None:
        return (
            "Models not loaded", "", "", "", "",
            f"**Models are not loaded.** Run `python train_models.py` first.\n\n{LOAD_ERROR}",
        )

    if not short_description or not short_description.strip():
        return (
            "Please enter a problem description.",
            "", "", "", "",
            "Enter a problem description to see recommendations.",
        )

    try:
        result = engine.predict_ticket(
            short_description=short_description.strip(),
            ticket_type=ticket_type or "",
            # Retain the empty value for compatibility with the existing model.
            issue_request="",
            software_system=software_system or "",
        )

        recommendations_text = format_recommendations(
            result.get("recommendations")
        )

        confidence = result.get("routing_confidence", "")
        try:
            confidence_text = f"{float(confidence):.2f}"
        except (ValueError, TypeError):
            confidence_text = str(confidence)

        return (
            str(result.get("category", "Not available")),
            str(result.get("subcategory", "Not available")),
            str(result.get("assignment_group", "Not available")),
            confidence_text,
            str(result.get("routing_status", "Not available")),
            recommendations_text,
        )

    except Exception as e:
        return (
            "Analysis error", "", "", "", "",
            f"**Could not analyze the ticket.**\n\n`{str(e)}`",
        )


# --------------------------------------------------
# Gradio interface
# --------------------------------------------------

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Smart IT Helpdesk",
) as demo:

    gr.Markdown(
        """
        # Smart IT Helpdesk
        ### AI-Powered Ticket Classification and Solution Recommendation

        Describe your IT problem to predict its category, identify a suitable
        support team, and find relevant resolutions from previously resolved tickets.
        """
    )

    if LOAD_ERROR:
        gr.Markdown(
            f"""
            **Models are not loaded.**

            Run `python train_models.py` in your project terminal, then restart
            this application.

            Error details: `{LOAD_ERROR}`
            """
        )

    with gr.Row(equal_height=False):

        with gr.Column(scale=1):
            gr.Markdown("## Enter Ticket Details")

            short_description = gr.Textbox(
                label="Problem Description *",
                placeholder=(
                    "Example: My laptop is connected to Wi-Fi "
                    "but I cannot access the internet."
                ),
                lines=4,
            )

            ticket_type = gr.Dropdown(
                choices=["Incident", "Request"],
                label="Ticket Type (optional)",
                value="Incident",
                allow_custom_value=True,
            )

            software_system = gr.Textbox(
                label="Software / System (optional)",
                placeholder="Example: Windows, Wi-Fi / Network, Microsoft Outlook",
                lines=2,
            )

            with gr.Row():
                analyze_button = gr.Button(
                    "Analyze Ticket",
                    variant="primary",
                )
                clear_button = gr.Button("Clear All")

        with gr.Column(scale=1):
            gr.Markdown("## Ticket Analysis")

            category_output = gr.Textbox(
                label="Predicted Category",
                interactive=False,
            )

            subcategory_output = gr.Textbox(
                label="Predicted Subcategory",
                interactive=False,
            )

            assignment_output = gr.Textbox(
                label="Recommended Assignment Group",
                interactive=False,
            )

            confidence_output = gr.Textbox(
                label="Routing Confidence",
                info="Based on historical routing data.",
                interactive=False,
            )

            status_output = gr.Textbox(
                label="Routing Status",
                interactive=False,
            )

    gr.Markdown("---")
    gr.Markdown("## Solution Recommendations")

    gr.Markdown(
        "Historical resolutions are suggestions from past tickets, not guaranteed fixes."
    )

    recommendations_output = gr.Markdown(
        value="Your recommended historical solutions will appear here after analysis."
    )

    analyze_button.click(
        fn=analyze_ticket,
        inputs=[
            short_description,
            ticket_type,
            software_system,
        ],
        outputs=[
            category_output,
            subcategory_output,
            assignment_output,
            confidence_output,
            status_output,
            recommendations_output,
        ],
    )

    clear_button.click(
        fn=lambda: (
            "", "Incident", "", "", "", "", "",
            "", "", "Your recommended historical solutions will appear here after analysis."
        ),
        inputs=[],
        outputs=[
            short_description,
            ticket_type,
            software_system,
            category_output,
            subcategory_output,
            assignment_output,
            confidence_output,
            status_output,
            recommendations_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()