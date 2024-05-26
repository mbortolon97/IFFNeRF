from sympy import symbols
import sympy

aabb_min = symbols('aabb_{min}')
aabb_max = symbols('aabb_{max}')

grid_size = symbols('grid')

grid_coords = symbols('coords')
samples = symbols('s')

aabbSize = aabb_max - aabb_min

norm_grid_coords = ((samples - aabb_min) * (1.0 / aabbSize * 2) - 1)  # interval between [-1., +1.]

grid_coords_eq = ((norm_grid_coords + 1) / 2) * (grid_size - 1)

grid_coords_simplify = sympy.simplify(grid_coords_eq)

sympy_solution = sympy.solve(sympy.Eq(grid_coords, grid_coords_simplify), samples)
print(sympy.simplify(sympy_solution[0]))