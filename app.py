# app.py
import os
import gradio as gr
from search import preload_many, run_search
from helper import shut_gpu

# Shut down GPU
shut_gpu()


# Build the Gradio UI layout and wire callbacks
def build_ui():
    # Use a simple clean theme
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # Title / header
        gr.Markdown("### 🔎 Image Search — Multi-Query & Multi-Metric (Deep / Color / UMAP)")

        # Two-column layout: controls on the left, results on the right
        with gr.Row():
            with gr.Column(scale=1):
                # Drag & drop 1–3 images (we’ll average query embeddings)
                img_in = gr.Files(file_types=["image"], label="Drop 1–3 query images", file_count="multiple")

                # Method selection (subset of {"deep","color","umap"})
                methods = gr.CheckboxGroup(
                    choices=["deep", "color", "umap"],
                    value=["deep"],
                    label="Methods"
                )

                # SQLite DB path (used to resolve image_id -> filepath)
                db_path = gr.Textbox(value="images.db", label="SQLite DB path")

                # Top-K final results after fusion
                topk = gr.Slider(1, 20, value=5, step=1, label="Top-K")

                # Per-method candidate pool before fusion (higher = better recall, slower)
                pool = gr.Slider(50, 1000, value=200, step=50, label="Candidate pool per method")

                # Fusion strategy (only mean or max)
                fuser = gr.Radio(choices=["mean", "max"], value="mean", label="Fusion")

                # Method weights (they will be normalized to sum to 1 over selected methods)
                gr.Markdown("**Weights (sum = 1 on selected methods):**")
                w_deep = gr.Slider(0, 1, value=1.0, step=0.05, label="w_deep")
                w_color = gr.Slider(0, 1, value=0.5, step=0.05, label="w_color")
                w_umap = gr.Slider(0, 1, value=0.5, step=0.05, label="w_umap")

                # Optional: trim long mount prefix from result captions
                trim_base = gr.Textbox(value="/Volumes/BigDataA", label="Trim prefix in captions")

                # Action buttons
                run_btn = gr.Button("Search", variant="primary")
                warm_btn = gr.Button("Preload now")

            with gr.Column(scale=2):
                # Image grid for query + top-K neighbors
                gallery = gr.Gallery(label="Results", columns=6, height=360, preview=False)

                # Numeric table for rank / id / score / path
                table = gr.Dataframe(
                    headers=["rank", "img_id", "fused_score", "filepath"],
                    datatype=["number", "number", "number", "str"],
                    interactive=False,
                    row_count=10,
                )

        # Pack inputs/outputs for callbacks
        inputs = [img_in, methods, db_path, topk, pool, fuser, w_deep, w_color, w_umap, trim_base]
        outputs = [gallery, table]

        # Trigger search automatically on file drop
        img_in.change(run_search, inputs=inputs, outputs=outputs)

        # Trigger search on button click
        run_btn.click(run_search, inputs=inputs, outputs=outputs)

        # Preload on app load (speeds up first query: loads models/indices)
        def _on_load(db):
            try:
                # Preload some common methods; adjust as you like
                preload_many(["deep", "umap"], db)
            except Exception:
                # Swallow preload errors to avoid breaking UI on start
                pass
            # No visible output
            return

        # Wire the on-load preload
        demo.load(_on_load, inputs=[db_path], outputs=[])

        # Manual preload (respect current method selection)
        def _manual_preload(method_list, db):
            try:
                chosen = [m for m in method_list if m in {"deep", "color", "umap"}] or ["deep"]
                preload_many(chosen, db)
            except Exception:
                pass
            # No visible output
            return

        # Wire the manual preload button
        warm_btn.click(_manual_preload, inputs=[methods, db_path], outputs=[])

    # Return the configured Gradio app
    return demo


# Run the app from CLI: python app.py
if __name__ == "__main__":
    # Build the UI
    demo = build_ui()

    # Launch the server; allow Gradio to read your external image root (for previews)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        allowed_paths=["/Volumes/BigDataA"],  # include your external disk root
    )