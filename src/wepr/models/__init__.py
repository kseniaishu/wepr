"""The models, kept deliberately few.

Three things are fitted, and each one can be explained in a sentence:

- `baselines` — day-of-year climatology and persistence. Not decoration: these are
  what every other number is reported against.
- `nextday` — a supervised model mapping today's observations to tomorrow's.
- `seasonal` — day-of-year climatology plus a fitted trend, for dates far enough
  out that no observations are available to condition on.

Preferring a small number of well-understood models over an ensemble is a
deliberate choice. An ensemble that cannot be explained is worth less here than a
ridge regression that can.
"""
