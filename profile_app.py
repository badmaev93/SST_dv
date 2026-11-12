import cProfile
import pstats
import app
import asyncio

# Setup fake request to _get_field_png
import numpy as np

# We need to mock the data because it's not precomputed
# Let's run precompute.py first!
