# Understanding Backpropagation — With Toy Examples

Backpropagation (short for *backward propagation of errors*) is the key algorithm that makes neural networks learn. If you’ve ever wondered **how weights in a neural net get updated**, this post will walk you through the intuition, math, and toy examples step by step.

---

## 🔹 1. What is Backpropagation?

At its core, backpropagation is about answering one question:

👉 *How should each weight in the network be adjusted to make predictions more accurate?*

It works in four steps:

1. **Forward pass** — pass input through the network to compute predictions.
2. **Loss computation** — measure how wrong the prediction is compared to the true output.
3. **Backward pass** — propagate the error backward through the network using the **chain rule of calculus** to compute gradients.
4. **Weight update** — adjust weights slightly using gradient descent.

Repeat this process thousands of times, and the network *learns*.

---

## 🔹 2. The Chain Rule

Backprop is built on the **chain rule**:

\[
\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}
\]

Neural networks are just *nested functions*. Backprop applies the chain rule to compute how the loss depends on each weight.

---

## 🔹 3. Toy Example: 1-Layer Network

Let’s start with the simplest case:

- Input: \(x = 2\)  
- Weight: \(w = 3\)  
- Target output: \(y = 10\)  
- Model:  
  \[
  \hat{y} = w \cdot x
  \]  
- Loss:  
  \[
  L = (y - \hat{y})^2
  \]

### Step 1. Forward Pass

\[
\hat{y} = 3 \cdot 2 = 6
\]  
\[
L = (10 - 6)^2 = 16
\]

### Step 2. Backward Pass

We want the gradient wrt \(w\):

\[
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w}
\]

- \(\frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y}) = -8\)  
- \(\frac{\partial \hat{y}}{\partial w} = x = 2\)  

So:

\[
\frac{\partial L}{\partial w} = -16
\]

### Step 3. Weight Update

With learning rate \(\eta = 0.1\):

\[
w_{\text{new}} = 3 - 0.1(-16) = 4.6
\]

### Step 4. New Forward Pass

\[
\hat{y} = 4.6 \cdot 2 = 9.2, \quad L = (10 - 9.2)^2 = 0.64
\]

✅ The loss went down. That’s backprop in action.

---

## 🔹 4. Extending to 2 Layers (With Sigmoid)

Now let’s add a hidden layer:

- Hidden layer:  
  \[
  h = \sigma(w_1 x), \quad \sigma(z) = \frac{1}{1+e^{-z}}
  \]
- Output layer:  
  \[
  \hat{y} = w_2 h
  \]
- Loss:  
  \[
  L = (y - \hat{y})^2
  \]

### Forward Pass

Let \(x=1, w_1=0.5, w_2=-1.0, y=0.5\).

1. \(z_1 = 0.5\)  
2. \(h = \sigma(0.5) \approx 0.622\)  
3. \(\hat{y} = -1.0 \cdot 0.622 = -0.622\)  
4. \(L = (0.5 - (-0.622))^2 \approx 1.259\)

### Backward Pass

- \(\frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y}) = -2.244\)  
- Gradient wrt \(w_2\):  
  \(\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial \hat{y}} \cdot h = -1.394\)  
- Gradient wrt \(w_1\):  
  \[
  \frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial \hat{y}} \cdot w_2 \cdot \sigma'(z_1) \cdot x
  \]
  where \(\sigma'(z_1) = \sigma(z_1)(1-\sigma(z_1)) \approx 0.235\).  
  So: \(0.527\).

### Update

With \(\eta=0.1\):

- \(w_2 \to -0.861\)  
- \(w_1 \to 0.447\)  

✅ Loss decreases from 1.259 → 1.052.

---

## 🔹 5. Using ReLU Instead

Suppose the hidden layer uses ReLU:

\[
h = \max(0, w_1 x)
\]

Derivative:

\[
\frac{d}{dz}\text{ReLU}(z) =
\begin{cases}
1 & z > 0 \\
0 & z < 0 \\
\text{undefined (set to 0 or 1)} & z = 0
\end{cases}
\]

So the gradient wrt \(w_1\) becomes:

\[
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial \hat{y}} \cdot w_2 \cdot \text{ReLU}'(z_1) \cdot x
\]

### Example

Take \(x=1, w_1=0.5, w_2=-1.0, y=0.5\).

- \(z_1 = 0.5 > 0 \Rightarrow h=0.5\)  
- \(\hat{y} = -0.5\), Loss = \(1.0\)  
- Gradient wrt \(w_1\):  
  \((-2)(0.5 - (-0.5)) \cdot (-1) \cdot 1 \cdot 1 = 2\)  

So \(w_1\) updates strongly.

But if \(z_1 < 0\), then \(\text{ReLU}'(z_1) = 0\), and **the gradient vanishes** — that neuron becomes “dead.”

---

## 🔹 6. Key Takeaways

- **Backprop = chain rule applied to neural nets.**  
- Each weight’s gradient measures how much it contributed to the error.  
- **Sigmoid:** smooth but small gradients → risk of vanishing.  
- **ReLU:** strong gradients when active, but zero when inactive → risk of dead neurons.  
- Weight updates gradually push predictions closer to targets.

---

## ✨ Closing Thought

Backpropagation is just calculus and the chain rule, applied carefully. Once you see it with toy examples, the magic of deep learning feels a lot more accessible.

