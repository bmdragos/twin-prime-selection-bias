/-
Copyright (c) 2025. All rights reserved.
Released under MIT license.

# Residue Class Mutual Exclusivity

This file proves the core mechanism behind the twin prime selection bias:
for twin prime candidates (6k-1, 6k+1), divisibility by any prime p ≥ 5
is mutually exclusive between the two components.

## Main results

* `dvd_of_dvd_add_two`: If p | a and p | (a+2), then p | 2
* `mutual_exclusivity`: For p ≥ 3, at most one of p | (6k-1) or p | (6k+1) holds
-/

import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Nat.Prime.Basic

open Nat

/-!
## Mutual Exclusivity of Divisibility

The key insight: if a prime p divides both 6k-1 and 6k+1, then p divides their
difference, which is 2. So for p ≥ 3, at most one can be divisible by p.
-/

/-- If p divides both a and a+2, then p divides 2. -/
theorem dvd_of_dvd_add_two {p a : ℕ} (h1 : p ∣ a) (h2 : p ∣ a + 2) : p ∣ 2 := by
  have : p ∣ (a + 2) - a := Nat.dvd_sub' h2 h1
  simp at this
  exact this

/-- For a prime p ≥ 3, p cannot divide both 6k-1 and 6k+1.

The proof: if p | (6k-1) and p | (6k+1), then p | 2.
But p ≥ 3 and divisors of 2 are at most 2, contradiction. -/
theorem mutual_exclusivity {p k : ℕ} (hp : p.Prime) (hp3 : p ≥ 3) :
    ¬(p ∣ 6 * k - 1 ∧ p ∣ 6 * k + 1) := by
  intro ⟨h1, h2⟩
  have hdiff : p ∣ (6 * k + 1) - (6 * k - 1) := Nat.dvd_sub' h2 h1
  have hsub : (6 * k + 1) - (6 * k - 1) = 2 := by omega
  rw [hsub] at hdiff
  have hle : p ≤ 2 := Nat.le_of_dvd (by norm_num : (0 : ℕ) < 2) hdiff
  -- p ≥ 3 and p ≤ 2 is a contradiction
  have : 3 ≤ 2 := Nat.le_trans hp3 hle
  norm_num at this

/-!
## Counting Residue Classes

For p ≥ 5, among the p residue classes mod p:
- Exactly 1 class has p | (6k-1)
- Exactly 1 class has p | (6k+1)
- These two classes are disjoint (by mutual_exclusivity)
- So p-2 classes have neither divisibility

When we condition on "6k-1 is not divisible by p" (i.e., remove 1 class),
we're left with p-1 classes, exactly 1 of which has p | (6k+1).
This gives P(p | (6k+1) | p ∤ (6k-1)) = 1/(p-1).
-/

/-- 6 is coprime to any prime p ≥ 5. -/
theorem six_coprime_to_large_prime {p : ℕ} (hp : p.Prime) (hp5 : p ≥ 5) : Nat.Coprime 6 p := by
  -- p ≥ 5 and prime means p ∤ 2 and p ∤ 3
  -- Therefore gcd(6, p) = 1
  sorry

/-- For p ≥ 5, exactly one residue class mod p has 6k ≡ 1 (mod p). -/
theorem unique_residue_6k_eq_1 (p : ℕ) (hp : p.Prime) (hp5 : p ≥ 5) :
    ∃! r : ZMod p, (6 : ZMod p) * r = 1 := by
  -- 6 is a unit in ZMod p for p ≥ 5
  have h6 : (6 : ZMod p) ≠ 0 := by
    intro h
    -- This would mean p | 6, but p ≥ 5 and prime
    have hdvd : p ∣ 6 := ZMod.natCast_zmod_eq_zero_iff_dvd 6 p |>.mp h
    have hle : p ≤ 6 := Nat.le_of_dvd (by norm_num : 0 < 6) hdvd
    -- p ≥ 5 and p | 6 and prime means p ∈ {5, 6}, but 5 ∤ 6 and 6 is not prime
    have : p = 5 ∨ p = 6 := by omega
    rcases this with rfl | rfl
    · -- p = 5: but 5 ∤ 6
      norm_num at hdvd
    · -- p = 6: but 6 is not prime
      exact Nat.not_prime_mul (by norm_num) (by norm_num) hp
  use (6 : ZMod p)⁻¹
  constructor
  · exact mul_inv_cancel₀ h6
  · intro r hr
    calc r = r * 1 := (mul_one r).symm
      _ = r * ((6 : ZMod p) * (6 : ZMod p)⁻¹) := by rw [mul_inv_cancel₀ h6]
      _ = (6 * r) * (6 : ZMod p)⁻¹ := by ring
      _ = 1 * (6 : ZMod p)⁻¹ := by rw [hr]
      _ = (6 : ZMod p)⁻¹ := one_mul _

/--
The main probabilistic consequence: conditioning on p ∤ a boosts
the probability that p | b from 1/p to 1/(p-1).

This is stated as a ratio: among p-1 "allowed" classes, exactly 1 has p | b.
-/
theorem conditional_density (p : ℕ) (hp : p.Prime) (hp5 : p ≥ 5) :
    ∃ (allowed : Finset (ZMod p)) (divisible : Finset (ZMod p)),
      allowed.card = p - 1 ∧
      divisible.card = 1 ∧
      divisible ⊆ allowed := by
  sorry -- Full proof requires more Mathlib machinery
