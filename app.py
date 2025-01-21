import gradio as gr
from scripts.yolo import train_yolo


def create_gradio_interface():
    with gr.Blocks() as app:
        # Title
        gr.Markdown("## YOLO Gradio App")

        with gr.Tab("Train"):
            gr.Markdown("## YOLO Train")

            # Input fields
            data_file_path = gr.Textbox(
                label="Enter Dataset Path (data.yaml)",
                placeholder="/path/to/data.yaml",
                lines=1,
            )

            # train_data_path = gr.Textbox(
            #     label="Enter Train Data Path",
            #     placeholder="/path/to/train/folder",
            #     lines=1,
            # )

            # test_data_path = gr.Textbox(
            #     label="Enter Test Data Path",
            #     placeholder="/path/to/test/folder",
            #     lines=1,
            # )

            # val_data_path = gr.Textbox(
            #     label="Enter Val Data Path",
            #     placeholder="/path/to/val/folder",
            #     lines=1,
            # )

            model_choice = gr.Dropdown(
                choices=[
                    "yolov5n",
                    "yolov5s",
                    "yolov5m",
                    "yolov5l",
                    "yolov5x",
                    "yolov8n",
                    "yolov8s",
                    "yolov8m",
                    "yolov8l",
                    "yolov8x",
                    "yolov9n",
                    "yolov9s",
                    "yolov9m",
                    "yolov9l",
                    "yolov9x",
                    "yolov10n",
                    "yolov10s",
                    "yolov10m",
                    "yolov10l",
                    "yolov10x",
                    "yolo11n",
                    "yolo11s",
                    "yolo11m",
                    "yolo11l",
                    "yolo11x",
                ],
                label="Select Model",
                value="yolo11s",
            )

            epochs = gr.Slider(
                minimum=1, maximum=100, label="Number of Epochs", value=1000, step=1
            )
            img_size = gr.Slider(
                minimum=320, maximum=1280, label="Image Size", value=640, step=1
            )
            batch_size = gr.Slider(
                minimum=1, maximum=64, label="Batch Size", value=16, step=1
            )

            # Output field
            output_text = gr.Textbox(label="Training Output")

            # Submit button
            submit_button = gr.Button("Train Model")

            # Define what happens on submit
            submit_button.click(
                fn=train_yolo,
                inputs=[model_choice, data_file_path, epochs, img_size, batch_size],
                outputs=output_text,
            )

        with gr.Tab("Validate"):
            gr.Markdown("## YOLO Validate")
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Flip")

        with gr.Tab("Predict"):
            gr.Markdown("## YOLO Predict")
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Flip")

        with gr.Tab("Export"):
            gr.Markdown("## YOLO Export")
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Flip")

        with gr.Tab("Track"):
            gr.Markdown("## YOLO Track")
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Flip")

        with gr.Tab("Benchmark"):
            gr.Markdown("## YOLO Benchmark")
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Flip")

    return app


demo = create_gradio_interface()

# Launch the app
demo.launch()
