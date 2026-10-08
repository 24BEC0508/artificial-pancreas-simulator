# Interactive Artificial Pancreas Dashboard

This adds `app.py` on top of the existing Bergman-model coursework project.

## Run

From this folder:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local address Streamlit prints, normally:

`http://localhost:8501`

On Windows, `run_app.bat` can also launch it.

## What the dashboard does

- Accepts custom meals with food name, quantity, calories, carbohydrates, sugar, protein and fat.
- Supports multiple meals at different simulated times.
- Uses the existing nonlinear Bergman model and existing controller implementations.
- Lets you choose PID, constrained PI with insulin-on-board, or LQR + observer.
- Includes CGM sampling, configurable noise and delay.
- Shows glucose, target, insulin delivery, plasma insulin and estimated insulin-on-board interactively.
- Supports single-step simulation or a live progressive simulation.

## Important modelling point

Calories and sugar are recorded as nutritional metadata. The Bergman meal disturbance is driven by **carbohydrate grams**, because the existing model represents meal glucose appearance from carbohydrate availability. The dashboard therefore requires carbohydrate input rather than pretending that calories alone determine glucose response.

This is a coursework simulation, not a medical device or treatment recommendation.
