"""Gradient attribution: which layers' attention drives computation toward -3/2?

Method:
1. Take baseline's first 50 tokens (teacher-forced, one forward pass)
2. Compute cos(h_L33[-1], B-moment_template)
3. Backprop to get gradient w.r.t. each layer's attention output
4. Gradient magnitude per layer = importance for reaching -3/2

Also decompose: attention contribution vs MLP contribution per layer.
"""
import json, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda"

SYS = ("You are solving an AMC 12A multiple choice math problem. "
       "Think step by step, show your work, then clearly state your "
       "final answer as (A), (B), (C), (D), or (E).")

P12_TEXT = (
    "The harmonic mean of a collection of numbers is the reciprocal of the "
    "arithmetic mean of the reciprocals of the numbers in the collection. "
    "For example, the harmonic mean of 4, 4, and 5 is\n\n"
    "1 / ((1/3)(1/4 + 1/4 + 1/5)) = 30/7.\n\n"
    "What is the harmonic mean of all the real roots of the 4050th degree "
    "polynomial\n\n"
    r"\prod_{k=1}^{2025} (kx^2 - 4x - 3) = "
    "(x^2 - 4x - 3)(2x^2 - 4x - 3)(3x^2 - 4x - 3)..."
    "(2025x^2 - 4x - 3)?\n\n"
    "(A) -5/3  (B) -3/2  (C) -6/5  (D) -5/6  (E) -2/3"
)
PROMPT = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{P12_TEXT}<|im_end|>\n<|im_start|>assistant\n"


def get_first_n_tokens(model, tokenizer, prompt, n=50):
    """Generate first N tokens greedily (for teacher forcing later)."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    gen_ids = []
    past_kv = None
    with torch.no_grad():
        for step in range(n):
            if step == 0:
                out = model(input_ids=input_ids, use_cache=True)
            else:
                out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids.append(next_id.item())
    del past_kv, out
    torch.cuda.empty_cache()
    return gen_ids


def main():
    print("=" * 70)
    print("GRADIENT ATTRIBUTION: Which layers drive computation toward -3/2?")
    print("=" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # fp16 fits in 12GB VRAM, gradients flow through sdpa backward
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE,
        trust_remote_code=True)
    model.eval()
    # Enable gradient computation + checkpointing to fit backward in 12GB
    for p in model.parameters():
        p.requires_grad_(True)
    model.gradient_checkpointing_enable()
    n_layers = len(model.model.layers)
    print(f"Loaded (float16, grad checkpointing). {n_layers} layers.\n", flush=True)

    # Load B-moment template from template matching experiment
    # We need to regenerate it since it was in float16
    # Instead: generate first 50 tokens, then do gradient attribution
    print("Generating baseline first 50 tokens...", flush=True)
    baseline_50 = get_first_n_tokens(model, tokenizer, PROMPT, n=50)
    text_50 = tokenizer.decode(baseline_50, skip_special_tokens=True)
    print(f"  First 50 tokens: {text_50[:100]}...\n", flush=True)

    # Build the full input: prompt + first 50 generated tokens
    prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids  # (1, prompt_len)
    gen_tensor = torch.tensor([baseline_50], dtype=torch.long)  # (1, 50)
    full_ids = torch.cat([prompt_ids, gen_tensor], dim=1).to(DEVICE)  # (1, prompt_len + 50)
    seq_len = full_ids.shape[1]
    print(f"Full sequence: {seq_len} tokens (prompt + 50 gen)\n", flush=True)

    # ===== FORWARD PASS WITH GRADIENT HOOKS =====
    # We'll capture the output of each layer's self_attn and mlp
    # Then compute gradient of cos(h_L33[-1], h_L33[-1]_deflated)
    # Since we don't have the deflated template in float32, we'll use
    # a proxy: compute gradient of the L33 hidden state norm projected
    # onto the "correct answer" direction.
    # Actually simpler: just measure gradient of final hidden state
    # at the last token w.r.t. each layer's attention output.

    # Register hooks to capture attention outputs with gradient
    attn_outputs = {}  # {layer: tensor}
    mlp_outputs = {}   # {layer: tensor}
    attn_hooks = []
    mlp_hooks = []

    def make_attn_hook(li):
        def hook(module, inp, output):
            # self_attn returns (attn_output, attn_weights, past_kv)
            # or just (attn_output,) depending on config
            attn_out = output[0]  # (batch, seq, d)
            attn_outputs[li] = attn_out
        return hook

    def make_mlp_hook(li):
        def hook(module, inp, output):
            mlp_outputs[li] = output  # (batch, seq, d)
        return hook

    for li in range(n_layers):
        h = model.model.layers[li].self_attn.register_forward_hook(make_attn_hook(li))
        attn_hooks.append(h)
        h = model.model.layers[li].mlp.register_forward_hook(make_mlp_hook(li))
        mlp_hooks.append(h)

    # Forward pass with gradients
    print("Running forward pass with gradient tracking...", flush=True)
    model.zero_grad()

    # Enable grad for embeddings to allow backprop
    out = model(input_ids=full_ids, use_cache=False, output_hidden_states=True)

    # Get hidden state at L33 (0-indexed: layer 33 output is hidden_states[34])
    # hidden_states[0] = embedding, hidden_states[i] = after layer i-1
    h_L33 = out.hidden_states[34][:, -1, :]  # (1, d) - last token at L33 output

    # Target: we want to maximize the projection onto the direction that
    # produces -3/2. Use the final logit for token B (id=33) as the scalar.
    # This directly measures "how much does each layer's attention push
    # the output toward generating B?"
    final_logits = out.logits[:, -1, :]  # (1, vocab)
    b_logit = final_logits[0, 33]  # scalar: logit for token "B"

    # Also compute relative: B logit minus mean of other answer logits
    other_logits = torch.stack([final_logits[0, i] for i in [32, 34, 35, 36]])
    b_advantage = b_logit - other_logits.mean()

    print(f"  B logit: {b_logit.item():.4f}")
    print(f"  B advantage over other answers: {b_advantage.item():.4f}")
    print(f"  Answer logits: A={final_logits[0,32].item():.3f} B={final_logits[0,33].item():.3f} "
          f"C={final_logits[0,34].item():.3f} D={final_logits[0,35].item():.3f} E={final_logits[0,36].item():.3f}")

    # Backprop from B logit
    print("\nBackpropagating from B logit...", flush=True)
    b_logit.backward(retain_graph=True)

    # Collect gradients at each layer's attention and MLP output
    attn_grad_norms = []
    mlp_grad_norms = []

    for li in range(n_layers):
        # Attention gradient: how much does this layer's attention output
        # at the last token affect the B logit?
        if li in attn_outputs and attn_outputs[li].grad is not None:
            g = attn_outputs[li].grad[0, -1, :]  # (d,) gradient at last token
            attn_grad_norms.append(g.norm().item())
        else:
            attn_grad_norms.append(0.0)

        if li in mlp_outputs and mlp_outputs[li].grad is not None:
            g = mlp_outputs[li].grad[0, -1, :]  # (d,)
            mlp_grad_norms.append(g.norm().item())
        else:
            mlp_grad_norms.append(0.0)

    # Clean up hooks
    for h in attn_hooks + mlp_hooks:
        h.remove()

    # ===== ALTERNATIVE: use retain_grad on captured tensors =====
    # The above might not work because hook outputs don't automatically
    # get gradients. Let me try a different approach: register hooks that
    # capture the gradient FLOWING THROUGH each layer.

    # Actually, let's use the hidden_states directly.
    # hidden_states[i] is the output AFTER layer i-1.
    # The gradient of b_logit w.r.t. hidden_states[i] at the last token
    # tells us how much that layer's output matters.

    print("\nUsing hidden_states gradient (more reliable)...", flush=True)

    # We need to recompute with retain_grad on hidden states
    # Clean up
    del out
    torch.cuda.empty_cache()
    model.zero_grad()
    attn_outputs.clear()
    mlp_outputs.clear()

    # Re-run with hidden states that have retain_grad
    # Use a manual approach: hook into each layer to retain grad on its output
    layer_outputs = {}
    layer_hooks = []

    def make_layer_hook(li):
        def hook(module, inp, output):
            # Decoder layer output: (hidden_states, ...)
            hs = output[0]
            hs.retain_grad()
            layer_outputs[li] = hs
        return hook

    for li in range(n_layers):
        h = model.model.layers[li].register_forward_hook(make_layer_hook(li))
        layer_hooks.append(h)

    # Forward
    out = model(input_ids=full_ids, use_cache=False)
    final_logits = out.logits[:, -1, :]
    b_logit = final_logits[0, 33]

    # Backward
    b_logit.backward()

    # Collect gradient norms at last token position for each layer
    layer_grad_norms = []
    for li in range(n_layers):
        if li in layer_outputs and layer_outputs[li].grad is not None:
            g = layer_outputs[li].grad[0, -1, :]
            layer_grad_norms.append(g.norm().item())
        else:
            layer_grad_norms.append(0.0)

    for h in layer_hooks:
        h.remove()

    # ===== DISPLAY =====
    print("\n" + "=" * 70)
    print("GRADIENT OF B-LOGIT w.r.t. LAYER OUTPUT (last token)")
    print("Higher = this layer's output more strongly influences B prediction")
    print("=" * 70)

    max_grad = max(layer_grad_norms) if layer_grad_norms else 1
    for li in range(n_layers):
        gn = layer_grad_norms[li]
        bar = "#" * int(40 * gn / max_grad) if max_grad > 0 else ""
        # Mark deflation-effective range
        marker = ""
        if 20 <= li <= 33:
            marker = " [COMPUTE]"
        elif li >= 34:
            marker = " [READ HEAD]"
        print(f"  L{li:>2d}: {gn:>10.6f}  {bar}{marker}")

    # Summary
    compute_total = sum(layer_grad_norms[20:34])
    readhead_total = sum(layer_grad_norms[34:36])
    early_total = sum(layer_grad_norms[0:20])
    total = sum(layer_grad_norms)

    print(f"\nGradient mass distribution:")
    print(f"  Early (L0-L19):    {early_total:>10.6f}  ({100*early_total/total:.1f}%)")
    print(f"  Compute (L20-L33): {compute_total:>10.6f}  ({100*compute_total/total:.1f}%)")
    print(f"  Read head (L34-35):{readhead_total:>10.6f}  ({100*readhead_total/total:.1f}%)")

    # Peak layers
    sorted_layers = sorted(range(n_layers), key=lambda i: layer_grad_norms[i], reverse=True)
    print(f"\nTop 10 layers by gradient magnitude:")
    for i, li in enumerate(sorted_layers[:10]):
        print(f"  #{i+1}: L{li} = {layer_grad_norms[li]:.6f}")

    # Save
    results = {
        "b_logit": b_logit.item(),
        "answer_logits": {"A": final_logits[0,32].item(), "B": final_logits[0,33].item(),
                          "C": final_logits[0,34].item(), "D": final_logits[0,35].item(),
                          "E": final_logits[0,36].item()},
        "layer_grad_norms": layer_grad_norms,
        "gradient_mass": {"early_L0_L19": early_total, "compute_L20_L33": compute_total,
                          "readhead_L34_L35": readhead_total},
        "top10_layers": sorted_layers[:10],
        "first_50_tokens": text_50,
    }
    out_path = "output/exp_deflation_gradient_p12.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
