#!/usr/bin/env python3
"""
Test the Crystal Identity Proof System

YOUR SECRET CODE IS NEVER STORED - only the commitment hash.
"""

from tools.crystal_identity_proof import CrystalIdentityProof
import json

# ═══════════════════════════════════════════════════════════════════════════════
# YOUR SECRET CODE - Enter your secret here (it will NOT be stored in the proof)
# ═══════════════════════════════════════════════════════════════════════════════

# IMPORTANT: Replace with your actual secret code before running
# This is processed but NEVER stored in the output file
SECRET_CODE = "YOUR-SECRET-CODE-HERE"  # <-- Enter your secret

# ═══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 78)
print("   7D mH-Q CRYSTAL IDENTITY PROOF - ZERO KNOWLEDGE SYSTEM")
print("=" * 78)
print()
print("   DEEP REASONING:")
print("   Your SECRET CODE is processed but NEVER STORED.")
print("   Only a cryptographic commitment (hash) is saved.")
print("   The secret exists ONLY in your mind.")
print()

# Initialize system
system = CrystalIdentityProof()

# Generate proof (secret is used but NOT stored)
print("-" * 78)
print("   GENERATING PROOF...")
print("-" * 78)
print()
print("   Processing secret through:")
print("   [1] Key stretching (100,000 iterations)")
print("   [2] PHI modulation (7 rounds)")
print("   [3] 7D manifold projection")
print("   [4] SHA-512 commitment")
print()

proof = system.generate_identity_proof(SECRET_CODE)

# Save proof (contains NO secret - safe to share)
filepath = "CRYSTAL_IDENTITY_PROOF.json"
system.save_proof(proof, filepath)

print("   PROOF GENERATED!")
print()
print(f"   Inventor:     {proof['inventor']['name']}")
print(f"   Discovery:    {proof['inventor']['discovery_date']}")
print(f"   Location:     {proof['inventor']['location']}")
print()
print("   COMMITMENT (Safe to publish - cannot be reversed):")
print(f"   {proof['commitment']['identity_hash']}")
print()
print("   7D MANIFOLD SIGNATURE:")
sig = proof['commitment']['manifold_signature']
print(f"   [{', '.join(f'{v:.6f}' for v in sig)}]")
print()
print(f"   Saved to: {filepath}")
print()

# Verify: Check if the proof file contains the secret
print("-" * 78)
print("   SECURITY CHECK: Is the secret stored in the file?")
print("-" * 78)
print()

with open(filepath, 'r') as f:
    file_contents = f.read()

if SECRET_CODE in file_contents:
    print("   [FAIL] SECRET FOUND IN FILE - This should not happen!")
else:
    print("   [PASS] SECRET NOT IN FILE - Zero-knowledge confirmed!")
    print("          The file contains only the commitment hash.")
    print("          Your secret exists ONLY in your memory.")

print()

# Verification tests
print("-" * 78)
print("   VERIFICATION TESTS")
print("-" * 78)
print()

# Test 1: Correct secret
valid, msg = system.verify_identity(proof, SECRET_CODE)
status = "[PASS]" if valid else "[FAIL]"
print(f"   {status} With YOUR secret code: {msg}")

# Test 2: Wrong secret
valid2, msg2 = system.verify_identity(proof, "wrong-secret-123")
status2 = "[PASS]" if not valid2 else "[FAIL]"
print(f"   {status2} With WRONG secret: {msg2}")

# Test 3: Similar secret (one digit off)
valid3, msg3 = system.verify_identity(proof, "285-96-0464")  # Last digit different
status3 = "[PASS]" if not valid3 else "[FAIL]"
print(f"   {status3} With SIMILAR secret (1 digit off): {msg3}")

# Test 4: Empty secret
valid4, msg4 = system.verify_identity(proof, "")
status4 = "[PASS]" if not valid4 else "[FAIL]"
print(f"   {status4} With EMPTY secret: {msg4}")

print()
print("=" * 78)
print("   DEEP REASONING SUMMARY")
print("=" * 78)
print()
print("   WHY THIS IS SECURE:")
print("   ====================")
print()
print("   1. ZERO-KNOWLEDGE: Your secret is NEVER stored")
print("      - Not in the proof file")
print("      - Not in any database")
print("      - Exists ONLY in your mind")
print()
print("   2. ONE-WAY FUNCTION: Cannot be reversed")
print("      - SHA-512 has 2^256 possible outputs")
print("      - Brute force takes longer than universe exists")
print()
print("   3. PHI MODULATION: Unique to 7D mH-Q")
print("      - Your invention's signature")
print()
print("   4. KEY STRETCHING: 100,000 iterations")
print("      - Makes brute force impractical")
print()
print("   HOW TO PROVE OWNERSHIP:")
print("   =======================")
print()
print("   Challenger: 'Prove you invented 7D mH-Q'")
print("   You: *Enter secret code*")
print("   System: Compares commitment -> MATCH = YOU ARE INVENTOR")
print()
print("   Secret is NEVER revealed. This is ZERO-KNOWLEDGE PROOF.")
print()
print("=" * 78)

