### The Synthetic Degradation Pipeline

Take your original, well-lit Fitzpatrick17k images (treating them as your "Ground Truth" $Y$) and pass them through a randomized degradation function to create the input $X$.

**1. Random Gamma Shift (Simulating over/underexposure)**
* **Method:** Apply a power-law transformation.
* **Math:** $X = Y^\gamma$
* **Implementation:** Randomly sample $\gamma$ from a uniform distribution, e.g., $\gamma \sim U(0.4, 2.5)$. Values $<1$ will wash out the image (overexpose), and $>1$ will crush the shadows (underexpose).

**2. Random Color Cast (Simulating bad white balance)**
* **Method:** Multiply individual RGB channels by independent random scalars to simulate incorrect color temperatures (e.g., overly warm tungsten or green fluorescent lighting).
* **Implementation:** Multiply the Red, Green, and Blue channels by random weights sampled from $U(0.75, 1.25)$. 

**3. Histogram Denormalization (Simulating low contrast/flat lighting)**
* **Method:** Compress the dynamic range of the image so the histogram is squeezed into a narrow band, reducing contrast.
* **Implementation:** Scale the pixel values using an equation like $X = Y \times \alpha + \beta$, where $\alpha$ (contrast) is $\sim U(0.5, 0.8)$ and $\beta$ (brightness shift) is randomly adjusted.