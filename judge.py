import json
import torch
from z3 import Solver, Int, Bool, Implies, And, sat

def z3_referee_reward(llm_response_str: str):
    """
    Evaluates the LLM's proposed JSON state against DMN and Z3 rules.
    Returns: (reward_score: float, feedback_message: str)
    """
    try:
        # 1. Parse the LLM's proposed action
        data = json.loads(llm_response_str)
        cost = data.get("cost", 0)
        is_weekend = data.get("is_weekend", False)
        eng_level = data.get("engineer_level", 1) # 1=Junior, 2=Mid, 3=Senior
        task_type = data.get("task_type", "maintenance")

        # 2. DMN Layer (Fast, static policy checks)
        # Example: Seniors cannot be scheduled for basic maintenance
        if task_type == "maintenance" and eng_level == 3:
            return -1.0, "DMN Rule Violated: Senior engineers cannot do basic maintenance."

        # 3. Z3 Referee Layer (Complex interdependent constraints)
        s = Solver()

        # Define Z3 variables
        z_cost = Int('cost')
        z_is_weekend = Bool('is_weekend')
        z_eng_level = Int('engineer_level')

        # Bind LLM outputs to the Z3 variables
        s.add(z_cost == cost)
        s.add(z_is_weekend == is_weekend)
        s.add(z_eng_level == eng_level)

        # --- Define Business Rulebook (with tracking) ---
        
        # Rule A: Hard budget limit
        s.assert_and_track(z_cost <= 2000, "budget_limit_exceeded")

        # Rule B: Weekend rate logic (If weekend + Senior, minimum cost is $1500)
        s.assert_and_track(
            Implies(And(z_is_weekend, z_eng_level == 3), z_cost >= 1500), 
            "invalid_weekend_senior_rate"
        )

        # Rule C: Juniors cannot work weekends
        s.assert_and_track(
            Implies(z_is_weekend, z_eng_level > 1), 
            "junior_weekend_restriction"
        )

        # 4. Evaluate the State
        result = s.check()

        if result == sat:
            # The LLM generated a perfectly valid, rule-compliant state!
            return +1.0, "Success: All constraints satisfied."
        
        else:
            # 5. Extract the Unsat Core for dynamic penalties
            core = s.unsat_core()
            failed_rules = [str(rule) for rule in core]
            
            # Dynamic Penalty: -0.5 for each rule broken. 
            # This teaches the model to break fewer rules over time.
            penalty = -0.5 * len(failed_rules)
            feedback = f"Z3 Constraint Violated: {', '.join(failed_rules)}"
            
            return penalty, feedback

    except json.JSONDecodeError:
        # Heavily penalize the model for failing to output valid JSON
        return -2.0, "Fatal Error: Output must be valid JSON."
    except Exception as e:
        return -1.0, f"System Error: {str(e)}"
