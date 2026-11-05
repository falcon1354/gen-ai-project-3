# Task 7 – Analysis

Fill this in after training (also mirror the answers in the Medium blog).

## 7.1 Reward-model accuracy

- Validation pairwise accuracy: **____%** (chance = 50%)
- Kinds of pairs it gets wrong:
- Three failure examples (prompt + chosen + rejected + scores):

### Example 1
### Example 2
### Example 3

## 7.2 PPO curves

- Did mean rφ rise over iterations?
- Did summary length change?
- Is a higher reward-model score the same as a better summary? (use human eval)

## 7.3 Role of β_KL

Explain using Eq. 41 what happens if β_KL = 0 (do **not** run that experiment).

## 7.4 Clip fraction

What did the clip fraction look like during training, and what does it say about update size?

## 7.5 DPO and the missing reward

Using Eqs. 55–58: where did the reward go, and why is Z(x) not needed?

Compare DPO implicit-reward accuracy vs reward-model pairwise accuracy on the same 1,000 val pairs:

| Method | Accuracy |
|--------|----------|
| Reward model | |
| DPO implicit | |

## 7.6 Memory, time, choice

| Stage | Models on GPU | Wall time (T4) |
|-------|----------------|----------------|
| SFT | | |
| Reward | | |
| PPO | | |
| DPO | | |

Which method would you pick for a new project, and why?
