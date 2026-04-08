# Vector Logic: The Geometry of "Michael"

During your demo, someone might ask: "If the model uses math, how does it know the difference between Michael Jordan and Michael Jackson?" 

The answer lies in **Orthogonality** (the 90-degree rule) and **Displacement**.

## 1. The 90-Degree Rule (Independence)
In high-dimensional vector space (like Granite's 768D), **90 degrees means "Unrelated."** 

If two concepts are 90 degrees apart, they are **Orthogonal**. This is a feature, not a bug. The model intentionally keeps the "Sports" dimension and the "Music" dimension at 90 degrees to each other so that data about Basketball doesn't "pollute" data about Pop Music.

## 2. The "Michael" Disambiguation
Think of the word **"Michael"** as a starting point at the center of a map. By itself, it is vague.

### The Jordan Path (The Sports Lane)
1. **"Michael"**: Starting point.
2. **"Jordan"**: Adds a vector pointing towards **Sports**.
3. **"Basketball"**: Moves further down that same line.
4. **"Allstar"**: In this "Sports Lane," Allstar means "NBA MVP."

### The Jackson Path (The Music Lane)
1. **"Michael"**: Same starting point.
2. **"Jackson"**: Adds a vector pointing **90 degrees away** from the Sports Lane, towards **Music**.
3. **"Thriller"**: Moves further down that Music line.
4. **"Allstar"**: In this "Music Lane," Allstar means "Grammy Winner."

## 3. How Inferences are Made
The model makes inferences by looking at the **Sum of Vectors**.

- `Michael + Jordan` = A vector pointing into the "Sports Legend" neighborhood.
- `Michael + Jackson` = A vector pointing into the "Music Legend" neighborhood.

Because those neighborhoods are **Orthogonal (90 degrees apart)**, the model never gets confused. It uses the "Sports" dimensions to look up Jordan and the "Music" dimensions to look up Jackson. 

## 4. The "Concept Arithmetic" (The Real Mic Drop)
The most famous example of this vector logic is:
**King - Man + Woman = Queen**

1. Take the vector for **King**.
2. Subtract the "concept" of **Manhood** (moving 90 degrees away from the male axis).
3. Add the "concept" of **Womanhood** (moving 90 degrees toward the female axis).
4. **The Result:** You land exactly at the coordinates for **Queen**.

## Summary for the Demo
"When you ask our Granite model a question, it isn't 'guessing.' It is performing high-speed vector addition. It navigates these 90-degree 'lanes' of human knowledge to ensure that when you ask about network errors, it doesn't give you instructions for a toaster—even if they both use the word 'power'."
