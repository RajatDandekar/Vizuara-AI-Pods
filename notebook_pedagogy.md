# Vizuara Google Colab Notebook — Pedagogical Specification

## Core Philosophy

Every notebook should feel like a **guided cognitive journey**, not a script dump.
The student should feel like they are discovering concepts alongside a mentor who
knows exactly when to explain, when to let them struggle, and when to reward them
with a satisfying result.

---

## 1. Learning Journey Structure

Organize every notebook in this exact sequence:

1. **Motivation and Big-Picture Overview**
   - Why does this concept matter?
   - What real-world problem does it solve?
   - What will the student be able to do by the end of this notebook?
   - Show a teaser of the final output (e.g., a generated image, a trained agent)

2. **Intuition Building (NO CODE initially)**
   - Use analogies, diagrams, and thought experiments
   - Connect to concepts students already know
   - Build mental models before formalizing anything

3. **Mathematical Grounding**
   - Present equations with full LaTeX rendering
   - After EVERY equation, explain what it means computationally:
     "This equation says: take the dot product of Q and K, scale it, 
     apply softmax, and multiply by V. Computationally, this means..."
   - Connect math to the intuition built in step 2

4. **Incremental Implementation (Modular Components)**
   - Build the system piece by piece
   - Each component should be testable independently
   - Show intermediate outputs at every step
   - Keep code minimal and readable — no unnecessary abstractions

5. **Student Fill-in-the-Blank Sections (TODO)**
   - Strategic gaps where students must complete code
   - Provide enough scaffolding that they know WHAT to implement
   - Provide a verification cell after each TODO so they know if they got it right
   - Include hints (collapsed/hidden if possible)

6. **Frequent Visualization and Feedback Checkpoints**
   - After every major step, visualize something
   - Students should see intermediate outputs constantly
   - Use matplotlib, plotly, or simple print statements
   - Never let more than 3-4 code cells pass without a visual checkpoint

7. **Final Working System**
   - Everything comes together into a tangible output
   - The output must be VISUALLY SATISFYING (generated images, animated plots, etc.)
   - The student should feel a sense of completion and ownership

8. **Reflection Questions and Extension Exercises**
   - "What would happen if we doubled the learning rate?"
   - "How would this architecture change for 3D data?"
   - "Try modifying X and observe what happens"
   - Optional challenge sections for advanced students

---

## 2. Educational Style Constraints

- **Do NOT spoon-feed everything.** Leave strategic gaps.
- After important equations, ALWAYS explain what they mean computationally.
- Every major section must connect theory → implementation.
- Keep code minimal and readable.
- Avoid unnecessary abstractions (no deep class hierarchies for teaching).
- Use simple architectures unless complexity is conceptually required.
- **Never use high-level libraries that hide core mechanics.**
  - Build forward passes manually
  - Write training loops explicitly
  - Implement loss functions from scratch
  - Avoid pre-built pipelines (e.g., no HuggingFace Diffusers for teaching diffusion)

---

## 3. Student Engagement Patterns

### Reflective Questions (inside markdown cells)
```markdown
### 🤔 Think About This
Before we implement the attention mechanism, ask yourself:
- Why do we need THREE matrices (Q, K, V) instead of just one?
- What would happen if we removed the scaling factor?
```

### Think-Before-Scrolling Prompts
```markdown
---
### ✋ Stop and Think
Before scrolling down to see the solution, try to:
1. Write down what shape the output tensor should be
2. Identify which dimension the softmax should be applied over
3. Predict what the attention weights will look like for this input

*Take 2 minutes. Then scroll down.*

---
```

### TODO Sections
```python
def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Query tensor of shape (batch, seq_len, d_k)
        K: Key tensor of shape (batch, seq_len, d_k)  
        V: Value tensor of shape (batch, seq_len, d_v)
    
    Returns:
        Attention output of shape (batch, seq_len, d_v)
    """
    d_k = Q.size(-1)
    
    # ============ TODO ============
    # Step 1: Compute attention scores (Q @ K^T)
    # Step 2: Scale by sqrt(d_k)
    # Step 3: Apply softmax over the last dimension
    # Step 4: Multiply by V
    # ==============================
    
    scores = ???  # YOUR CODE HERE
    
    return output
```

### Verification Cells
```python
# ✅ Verification: Run this cell to check your implementation
expected_output = torch.tensor([...])
your_output = scaled_dot_product_attention(test_Q, test_K, test_V)
assert torch.allclose(your_output, expected_output, atol=1e-4), \
    f"❌ Output doesn't match. Expected shape {expected_output.shape}, got {your_output.shape}"
print("✅ Correct! Your attention implementation works perfectly.")
```

### Visualization Checkpoints
```python
# 📊 Let's visualize what just happened
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(attention_weights[0].detach().numpy(), cmap='viridis')
plt.title("Attention Weights")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(output[0].detach().numpy(), cmap='coolwarm')
plt.title("Output Embeddings")
plt.colorbar()
plt.tight_layout()
plt.show()
```

---

## 4. Notebook Formatting Requirements

### Cell Types and Order
- Alternate between markdown and code cells naturally
- Never have more than 2 consecutive code cells without a markdown explanation
- Never have more than 2 consecutive markdown cells without code or a visual

### Section Headings
```markdown
# 🚀 Notebook Title: Building X from Scratch

## 1. Why Does This Matter?
## 2. Building Intuition  
## 3. The Mathematics
## 4. Let's Build It — Component by Component
### 4.1 Component A
### 4.2 Component B
## 5. Putting It All Together
## 6. Training and Results
## 7. Your Turn — Experiments
## 8. Reflection and Next Steps
```

### Emoji Usage (sparingly, for visual anchors)
- 🚀 Main title
- 🤔 Reflection/thinking prompts
- ✋ Stop and think
- ✅ Verification/success
- 📊 Visualization checkpoints
- 🔧 TODO/implementation sections
- 💡 Key insights
- ⚠️ Common pitfalls/warnings
- 🎯 Final output/deliverable

### Code Style
- Clear variable names (no single letters except standard math notation)
- Comments explaining WHY, not WHAT
- Type hints where helpful
- Docstrings for all functions
- Max ~30 lines per code cell (break up long implementations)

### Colab-Specific
- Start with a GPU check / setup cell
- Install any required packages with `!pip install`
- Use `torch` as the primary framework
- Include `%matplotlib inline` where needed
- Set random seeds for reproducibility

---

## 5. Grounding in Fundamentals

The golden rule: **If a library hides the core mechanic being taught, implement it manually.**

Examples:
- Teaching attention? Implement Q, K, V projections and scaled dot-product manually.
- Teaching diffusion? Implement forward noising, beta schedules, and sampling loops manually.
- Teaching world models? Implement the VAE encoder, RNN dynamics, and imagination loop manually.
- Teaching RL? Implement the policy gradient or Q-learning update manually.

Only use high-level libraries for things that are NOT the focus of the lesson:
- Use torchvision for loading datasets (not the focus)
- Use matplotlib for plotting (not the focus)
- Use torch.nn for basic layers like Linear, Conv2d (building blocks, not the concept)

---

## 6. Final Output Requirements

Every notebook MUST produce a tangible, visually satisfying final output:
- World Models → agent playing inside its own dream (rendered frames)
- Diffusion Models → grid of generated images from noise
- Attention → visualization of attention patterns on real text
- RL → reward curve + video/frames of trained agent
- JEPA → visualization of learned representations
- VLA → simulated robot executing a command

The student should be able to screenshot the final output and feel proud.