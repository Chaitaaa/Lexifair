import numpy as np
import networkx as nx
import pulp
from scipy.optimize import linear_sum_assignment

# Cost matrix (tasks × agents)
cost_matrix = np.array([
    [9, 5, 6],
    [8, 2, 4],
    [7, 3, 1]
])

m, n = cost_matrix.shape  # m tasks, n agents

efficient_cost = 1  # Initialize efficient cost

def efficient_assignment(cost_matrix):
    # Use the Hungarian algorithm to get the optimal one-to-one assignment minimizing total cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    assignment = np.zeros_like(cost_matrix, dtype=int)
    for r, c in zip(row_ind, col_ind):
        assignment[r, c] = 1

    total_cost = cost_matrix[row_ind, col_ind].sum()
    agent_costs = [cost_matrix[:, j][assignment[:, j] == 1].sum() for j in range(n)]
    sorted_costs = sorted(agent_costs, reverse=True)

    global efficient_cost
    efficient_cost = total_cost  # Store efficient cost after Min-Max solution

    return {
        "Assignment Matrix (Efficient)": assignment,
        "Agent Costs": agent_costs,
        "Sorted Costs": sorted_costs,
        "Total Cost": total_cost
    }

def min_max_fair_assignment(cost_matrix):
    # Phase 1: Minimize max individual cost
    prob1 = pulp.LpProblem("MinMaxPhase1", pulp.LpMinimize)
    x1 = [[pulp.LpVariable(f"x1_{i}_{j}", cat='Binary') for j in range(n)] for i in range(m)]
    z1 = pulp.LpVariable("z1")

    prob1 += z1
    for i in range(m):
        prob1 += pulp.lpSum(x1[i][j] for j in range(n)) == 1
    for j in range(n):
        prob1 += pulp.lpSum(x1[i][j] for i in range(m)) == 1
        prob1 += pulp.lpSum(cost_matrix[i][j] * x1[i][j] for i in range(m)) <= z1

    prob1.solve(pulp.PULP_CBC_CMD(msg=False))
    z_opt = pulp.value(z1)

    # Phase 2: Minimize total cost, subject to max cost constraint from phase 1
    prob2 = pulp.LpProblem("MinTotalCostWithZBound", pulp.LpMinimize)
    x2 = [[pulp.LpVariable(f"x2_{i}_{j}", cat='Binary') for j in range(n)] for i in range(m)]

    total_cost_expr = pulp.lpSum(cost_matrix[i][j] * x2[i][j] for i in range(m) for j in range(n))
    prob2 += total_cost_expr

    for i in range(m):
        prob2 += pulp.lpSum(x2[i][j] for j in range(n)) == 1
    for j in range(n):
        prob2 += pulp.lpSum(x2[i][j] for i in range(m)) == 1
        prob2 += pulp.lpSum(cost_matrix[i][j] * x2[i][j] for i in range(m)) <= z_opt

    prob2.solve(pulp.PULP_CBC_CMD(msg=False))

    assignment = np.zeros((m, n), dtype=int)
    for i in range(m):
        for j in range(n):
            if x2[i][j].varValue == 1:
                assignment[i][j] = 1

    agent_costs = [sum(cost_matrix[i][j] for i in range(m) if assignment[i][j]) for j in range(n)]
    total_cost = sum(cost_matrix[i][j] for i in range(m) for j in range(n) if assignment[i][j])
    sorted_costs = sorted(agent_costs, reverse=True)

    POF = (total_cost - efficient_cost) / efficient_cost if efficient_cost != 0 else 0

    

    return {
        "Assignment Matrix (Min-Max)": assignment,
        "Agent Costs": agent_costs,
        "Sorted Costs": sorted_costs,
        "Total Cost": total_cost,
        "Max Individual Cost": z_opt,
        "Price of Fairness (POF)": round(POF, 4)
    }



def network_flow_lexifair(cost_matrix):
    assert m == n, "For one-to-one matching, number of tasks and agents must be equal."

    G_template = nx.DiGraph()
    source = "s"
    sink = "t"

    for task in range(m):
        G_template.add_edge(source, f"task_{task}", capacity=1)
    for agent in range(n):
        G_template.add_edge(f"agent_{agent}", sink, capacity=1)

    edges = [(i, j, cost_matrix[i][j]) for i in range(m) for j in range(n)]
    edges.sort(key=lambda x: x[2])  # Increasing order of cost

    assignment = np.zeros((m, n), dtype=int)
    used_edges = set()
    flow_dict = None

    # Step through edges, adding lowest-cost edges first
    for i, j, cost in edges:
        G_template.add_edge(f"task_{i}", f"agent_{j}", capacity=1)
        
        # Try flow on the current graph
        flow_value, current_flow = nx.maximum_flow(G_template, source, sink)

        if flow_value == m:
            flow_dict = current_flow
            break

    if flow_dict is None:
        print(" No valid lexifair assignment found.")
        return None

    for task in range(m):
        task_node = f"task_{task}"
        for agent in range(n):
            agent_node = f"agent_{agent}"
            if flow_dict[task_node].get(agent_node, 0) == 1:
                assignment[task][agent] = 1

    # Fix agent cost calculation
    agent_costs = [sum(cost_matrix[i][j] for i in range(m) if assignment[i][j]) for j in range(n)]
    total_cost = sum(cost_matrix[i][j] for i in range(m) for j in range(n) if assignment[i][j])
    sorted_costs = sorted(agent_costs, reverse=True)

    POF = (total_cost - efficient_cost) / efficient_cost if efficient_cost != 0 else 0

    return {
        "Assignment Matrix (Lexifair)": assignment,
        "Agent Costs": agent_costs,
        "Sorted Costs": sorted_costs,
        "Total Cost": total_cost,
        "Max Individual Cost": max(agent_costs),
        "Price of Fairness (POF)": round(POF, 4)
    }


# 🎯 Run and compare
print("----- Efficient Assignment -----")
efficient_results = efficient_assignment(cost_matrix)
for k, v in efficient_results.items():
    print(f"{k}: \n{v}\n")

print("----- Min-Max Fairness -----")
minmax_results = min_max_fair_assignment(cost_matrix)
for k, v in minmax_results.items():
    print(f"{k}: \n{v}\n")

print("----- Lexifairness (Network Flow) -----")
lexifair_results = network_flow_lexifair(cost_matrix)
if lexifair_results:
    for k, v in lexifair_results.items():
        print(f"{k}: \n{v}\n")
