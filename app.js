// 3. Import the pipeline and env objects
import { pipeline, env } from '@xenova/transformers';
// --- Configuration ---
// In a real app, you'd host the .wasm files yourself.
// For this demo, we'll allow remote models.
env.allowRemoteModels = true;
// Use the browser's cache for model files
env.useBrowserCache = true;
// Skip the WASM proxy for simpler setup
env.backends.onnx.wasm.proxy = false;
// --- Get UI Elements ---
const status = document.getElementById('status');
const promptInput = document.getElementById('prompt-input');
const generateBtn = document.getElementById('generate-btn');
const output = document.getElementById('output');
// --- The Core Logic ---
(async function() {
    try {
        // 4. Create a text-generation pipeline
        // This is the magic line. We're loading a 4-bit quantized Phi-3 model
        // from Hugging Face, optimized by Xenova.
        status.textContent = 'Loading model (Phi-3-mini-4k-instruct_q4)... This can take a minute on first load.';
        
        const generator = await pipeline('text-generation', 'Xenova/Phi-3-mini-4k-instruct_q4', {
            quantized: true,
            device: 'webgpu', // Request WebGPU backend
            dtype: 'q4'        // Specify 4-bit quantization
        });
        status.textContent = 'Model loaded. Ready to chat!';
        generateBtn.disabled = false;
        // 5. Set up the button click event
        generateBtn.addEventListener('click', async () => {
            output.textContent = 'Generating...';
            generateBtn.disabled = true;
            const prompt = promptInput.value;
            
            // The prompt format for Phi-3 is specific
            const messages = [
                { role: 'user', content: prompt }
            ];
            // 6. Run the generator!
            // All computation happens here, on the user's device.
            const result = await generator(messages, {
                max_new_tokens: 512,  // Control output length
                temperature: 0.7,
                top_k: 50,
            });
            // Extract the generated text from the complex result object
            const generatedContent = result[0].generated_text[result[0].generated_text.length - 1].content;
            output.textContent = generatedContent;
            
            generateBtn.disabled = false;
        });
    } catch (err) {
        status.textContent = `Error: ${err.message}`;
        console.error(err);
    }
})();
