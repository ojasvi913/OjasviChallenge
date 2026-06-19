import gradio as gr
import subprocess
import os
import tempfile
import shutil

def run_pipeline(input_file):
    if input_file is None:
        return "Please upload a candidates.jsonl file.", None

    # Determine output path
    output_csv = "final_rankings.csv"

    # Use the uploaded file path
    input_path = input_file.name

    # Run the pipeline script
    command = ["python", "run_pipeline.py", "--candidates", input_path, "--out", output_csv]
    
    try:
        # Run subprocess
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            return f"Pipeline finished successfully!\n\nLogs:\n{result.stdout}", output_csv
        else:
            return f"Error running pipeline.\n\nError logs:\n{result.stderr}\n\nStandard logs:\n{result.stdout}", None
    except Exception as e:
        return f"An exception occurred: {str(e)}", None

# Create Gradio interface
with gr.Blocks(title="AI Data Challenge Pipeline Sandbox") as demo:
    gr.Markdown("# 🏆 Redrob AI Data Challenge - Pipeline Sandbox")
    gr.Markdown("Upload your `candidates.jsonl` file to run the evaluation pipeline and generate rankings.")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload candidates.jsonl")
            run_btn = gr.Button("Run Pipeline", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="Execution Logs", lines=10)
            output_file = gr.File(label="Download final_rankings.csv")
            
    gr.Examples(
        examples=[["data/sample_100_candidates.jsonl"]],
        inputs=file_input,
        label="Click the sample below to pre-load a small dataset (meets Hackathon Section 10.5 constraint)"
    )

    run_btn.click(
        fn=run_pipeline,
        inputs=file_input,
        outputs=[output_text, output_file]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
