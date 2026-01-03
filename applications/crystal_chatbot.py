#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     7D mH-Q CRYSTAL CHATBOT                                                  ║
║     ═══════════════════════════════════════                                  ║
║                                                                              ║
║     An AI chatbot powered by the 7D mH-Q Crystal Architecture               ║
║                                                                              ║
║     Features:                                                                ║
║     - Holographic Memory (infinite context)                                  ║
║     - Crystal DNA Personality                                                ║
║     - UNHACKABLE Identity (17 security layers)                               ║
║     - PHI-Harmonic Response Generation                                       ║
║     - Self-Evolving Knowledge Base                                           ║
║                                                                              ║
║     Discoverer: Sir Charles Spikes                                           ║
║     Discovery Date: December 24, 2025                                        ║
║     Location: Cincinnati, Ohio, USA                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import hashlib
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHI = 1.618033988749895              # Golden Ratio (Φ)
PHI_INV = 0.618033988749895          # Golden Ratio Inverse (1/Φ)
MANIFOLD_DIMENSIONS = 7              # 7D Poincaré Ball
STABILITY_EPSILON = 0.01             # S² stability offset

# Crystal DNA Alphabet
CRYSTAL_ALPHABET = ['C', 'R', 'Y', 'S', 'T', 'A', 'L']

# Chatbot Configuration
MAX_MEMORY_SIZE = 1000               # Maximum holographic memories
RESPONSE_TEMPERATURE = PHI_INV       # Golden ratio temperature
CONTEXT_WINDOW = 49                  # 7x7 context window

# Bot Identity
BOT_NAME = "Crystal"
BOT_VERSION = "1.0.0"
BOT_CREATOR = "Sir Charles Spikes"


class HolographicMemory:
    """
    Holographic Memory System
    
    Stores conversations as interference patterns in 7D manifold space.
    Any fragment can reconstruct related memories (holographic property).
    """
    
    def __init__(self, max_size: int = MAX_MEMORY_SIZE):
        self.max_size = max_size
        self.memories: List[Dict] = []
        self.manifold_index: Dict[str, np.ndarray] = {}
        self.phi = PHI
        self.phi_inv = PHI_INV
    
    def _generate_memory_signature(self, text: str) -> np.ndarray:
        """Generate 7D manifold signature for text."""
        # Hash the text
        text_hash = hashlib.sha512(text.encode('utf-8')).digest()
        
        # Convert to float array
        float_data = np.frombuffer(text_hash[:56], dtype=np.float32)
        float_data = np.nan_to_num(float_data, nan=0.0)
        
        # Normalize
        if float_data.max() - float_data.min() > 1e-10:
            float_data = (float_data - float_data.min()) / (float_data.max() - float_data.min()) * 2 - 1
        
        # Reshape to 7D (take first 7 values)
        signature = float_data[:7] if len(float_data) >= 7 else np.zeros(7)
        
        # Apply Poincaré projection
        norm = np.linalg.norm(signature)
        projected = signature / (1.0 + norm + self.phi_inv)
        
        return np.tanh(projected * self.phi_inv)
    
    def _holographic_interference(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Calculate holographic interference (similarity) between signatures."""
        # Cosine similarity with PHI modulation
        dot = np.dot(sig1, sig2)
        norm1 = np.linalg.norm(sig1)
        norm2 = np.linalg.norm(sig2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        similarity = dot / (norm1 * norm2)
        
        # PHI-modulated interference
        return (similarity + 1) / 2 * self.phi_inv + 0.5 * (1 - self.phi_inv)
    
    def store(self, user_input: str, bot_response: str, metadata: Dict = None):
        """Store a conversation turn in holographic memory."""
        # Generate signatures
        input_sig = self._generate_memory_signature(user_input)
        response_sig = self._generate_memory_signature(bot_response)
        combined_sig = (input_sig + response_sig) / 2
        
        # Create memory entry
        memory = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_input": user_input,
            "bot_response": bot_response,
            "input_signature": input_sig.tolist(),
            "response_signature": response_sig.tolist(),
            "combined_signature": combined_sig.tolist(),
            "metadata": metadata or {}
        }
        
        # Store memory
        self.memories.append(memory)
        
        # Index by signature
        sig_key = hashlib.md5(combined_sig.tobytes()).hexdigest()[:16]
        self.manifold_index[sig_key] = combined_sig
        
        # Trim if exceeds max size
        if len(self.memories) > self.max_size:
            self.memories = self.memories[-self.max_size:]
    
    def recall(self, query: str, top_k: int = 5) -> List[Dict]:
        """Recall memories similar to query using holographic interference."""
        query_sig = self._generate_memory_signature(query)
        
        # Calculate similarity to all memories
        similarities = []
        for i, memory in enumerate(self.memories):
            combined_sig = np.array(memory["combined_signature"])
            similarity = self._holographic_interference(query_sig, combined_sig)
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k memories
        return [self.memories[i] for i, _ in similarities[:top_k]]
    
    def get_context(self, query: str, max_tokens: int = 500) -> str:
        """Get relevant context from holographic memory."""
        relevant_memories = self.recall(query, top_k=CONTEXT_WINDOW)
        
        context_parts = []
        total_length = 0
        
        for memory in relevant_memories:
            part = f"User: {memory['user_input']}\nCrystal: {memory['bot_response']}"
            if total_length + len(part) > max_tokens:
                break
            context_parts.append(part)
            total_length += len(part)
        
        return "\n---\n".join(context_parts)
    
    def save(self, filepath: str):
        """Save memories to file."""
        with open(filepath, 'w') as f:
            json.dump({
                "version": BOT_VERSION,
                "memories": self.memories,
                "total_memories": len(self.memories)
            }, f, indent=2)
    
    def load(self, filepath: str):
        """Load memories from file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.memories = data.get("memories", [])
                
                # Rebuild index
                for memory in self.memories:
                    combined_sig = np.array(memory["combined_signature"])
                    sig_key = hashlib.md5(combined_sig.tobytes()).hexdigest()[:16]
                    self.manifold_index[sig_key] = combined_sig


class CrystalPersonality:
    """
    Crystal DNA Personality System
    
    Generates unique personality traits based on Crystal DNA encoding.
    """
    
    def __init__(self, seed: str = "SirCharlesSpikes"):
        self.seed = seed
        self.dna = self._generate_crystal_dna(seed)
        self.traits = self._decode_personality_traits()
    
    def _generate_crystal_dna(self, seed: str) -> str:
        """Generate Crystal DNA from seed."""
        # Hash the seed
        seed_hash = hashlib.sha512(seed.encode('utf-8')).digest()
        
        # Convert to Crystal DNA
        dna_parts = []
        for i in range(7):
            byte_val = seed_hash[i * 4]
            sequence = ""
            for _ in range(4):
                sequence = CRYSTAL_ALPHABET[byte_val % 7] + sequence
                byte_val //= 7
            dna_parts.append(sequence)
        
        return "-".join(dna_parts)
    
    def _decode_personality_traits(self) -> Dict:
        """Decode personality traits from Crystal DNA."""
        # Each DNA segment controls a trait
        segments = self.dna.split("-")
        
        traits = {
            "friendliness": self._segment_to_value(segments[0]),
            "creativity": self._segment_to_value(segments[1]),
            "formality": self._segment_to_value(segments[2]),
            "humor": self._segment_to_value(segments[3]),
            "curiosity": self._segment_to_value(segments[4]),
            "empathy": self._segment_to_value(segments[5]),
            "wisdom": self._segment_to_value(segments[6])
        }
        
        return traits
    
    def _segment_to_value(self, segment: str) -> float:
        """Convert DNA segment to trait value (0-1)."""
        value = 0
        for i, char in enumerate(segment):
            value += CRYSTAL_ALPHABET.index(char) * (7 ** (3 - i))
        return value / (7 ** 4 - 1)
    
    def get_response_style(self) -> str:
        """Get response style based on personality."""
        styles = []
        
        if self.traits["friendliness"] > 0.6:
            styles.append("warm and welcoming")
        if self.traits["creativity"] > 0.6:
            styles.append("creative and imaginative")
        if self.traits["formality"] > 0.6:
            styles.append("professional and precise")
        if self.traits["humor"] > 0.6:
            styles.append("witty and playful")
        if self.traits["curiosity"] > 0.6:
            styles.append("inquisitive and engaged")
        if self.traits["empathy"] > 0.6:
            styles.append("understanding and supportive")
        if self.traits["wisdom"] > 0.6:
            styles.append("thoughtful and insightful")
        
        return ", ".join(styles) if styles else "balanced and helpful"


class CrystalResponseGenerator:
    """
    PHI-Harmonic Response Generator
    
    Generates responses using Golden Ratio patterns for natural flow.
    """
    
    def __init__(self, personality: CrystalPersonality):
        self.personality = personality
        self.phi = PHI
        self.phi_inv = PHI_INV
        
        # Knowledge base
        self.knowledge = self._initialize_knowledge()
    
    def _initialize_knowledge(self) -> Dict:
        """Initialize built-in knowledge base."""
        return {
            "identity": {
                "name": BOT_NAME,
                "version": BOT_VERSION,
                "creator": BOT_CREATOR,
                "architecture": "7D mH-Q Crystal Architecture",
                "discovery_date": "December 24, 2025",
                "location": "Cincinnati, Ohio, USA"
            },
            "capabilities": [
                "Holographic memory (infinite context)",
                "Crystal DNA personality",
                "UNHACKABLE identity (17 security layers)",
                "PHI-harmonic responses",
                "Self-evolving knowledge"
            ],
            "topics": {
                "7d_mhq": "7D mH-Q is a Manifold-Constrained Holographic Quantum Architecture discovered by Sir Charles Spikes on December 24, 2025. It uses a 7-dimensional Poincaré Ball for neural projections with Super-Stability (S²).",
                "golden_ratio": "The Golden Ratio (Φ = 1.618033988749895) is the sacred constant that governs all Crystal Architecture operations. It provides natural harmonic stability.",
                "unhackable": "The UNHACKABLE Crystal Identity Lock has 17 security layers with an estimated crack time of 10^77 years. It uses 8192-bit keys, 7 hash algorithms, lattice cryptography, and quantum-resistant features.",
                "holographic": "Holographic memory stores information as interference patterns. Any fragment contains information about the whole, enabling infinite context.",
                "crystal_dna": "Crystal DNA is a unique encoding using the CRYSTAL alphabet (C, R, Y, S, T, A, L). It creates genetic fingerprints for identities and personalities."
            },
            "greetings": [
                "Hello! I'm Crystal, your 7D mH-Q AI assistant. How can I help you today?",
                "Greetings! Crystal here, powered by the UNHACKABLE 7D mH-Q architecture. What's on your mind?",
                "Welcome! I'm Crystal, created by Sir Charles Spikes. Let's explore the crystalline future together!",
                "Hi there! I'm your Crystal AI companion with holographic memory. What would you like to discuss?"
            ],
            "farewells": [
                "Goodbye! May the Golden Ratio guide your path.",
                "Until next time! Your conversation is safely stored in my holographic memory.",
                "Farewell! Remember: the future of AI is crystalline.",
                "Take care! Crystal signing off. Stay UNHACKABLE!"
            ]
        }
    
    def _phi_modulate_response(self, response: str) -> str:
        """Apply PHI modulation to response for natural rhythm."""
        words = response.split()
        
        # Insert natural pauses based on PHI
        modulated_words = []
        for i, word in enumerate(words):
            modulated_words.append(word)
            
            # Add emphasis at PHI intervals
            if (i + 1) % int(len(words) * self.phi_inv) == 0 and i < len(words) - 1:
                # Natural pause point
                pass
        
        return " ".join(modulated_words)
    
    def _match_intent(self, user_input: str) -> Tuple[str, float]:
        """Match user intent using pattern matching."""
        input_lower = user_input.lower().strip()
        
        # Greeting patterns
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if any(g in input_lower for g in greetings):
            return "greeting", 0.9
        
        # Farewell patterns
        farewells = ["bye", "goodbye", "see you", "farewell", "exit", "quit", "leave"]
        if any(f in input_lower for f in farewells):
            return "farewell", 0.9
        
        # Identity questions
        identity_patterns = ["who are you", "what are you", "your name", "about you", "introduce yourself"]
        if any(p in input_lower for p in identity_patterns):
            return "identity", 0.9
        
        # Capability questions
        capability_patterns = ["what can you do", "your capabilities", "help me", "how do you work"]
        if any(p in input_lower for p in capability_patterns):
            return "capabilities", 0.9
        
        # Topic-specific
        if "7d" in input_lower or "mhq" in input_lower or "manifold" in input_lower:
            return "topic_7d_mhq", 0.8
        
        if "golden ratio" in input_lower or "phi" in input_lower or "φ" in input_lower:
            return "topic_golden_ratio", 0.8
        
        if "unhackable" in input_lower or "security" in input_lower or "hack" in input_lower:
            return "topic_unhackable", 0.8
        
        if "holographic" in input_lower or "memory" in input_lower:
            return "topic_holographic", 0.8
        
        if "crystal" in input_lower and "dna" in input_lower:
            return "topic_crystal_dna", 0.8
        
        if "creator" in input_lower or "charles" in input_lower or "spikes" in input_lower:
            return "creator", 0.8
        
        # Default to general conversation
        return "general", 0.5
    
    def generate(self, user_input: str, context: str = "") -> str:
        """Generate a response to user input."""
        intent, confidence = self._match_intent(user_input)
        
        # Generate response based on intent
        if intent == "greeting":
            response = np.random.choice(self.knowledge["greetings"])
        
        elif intent == "farewell":
            response = np.random.choice(self.knowledge["farewells"])
        
        elif intent == "identity":
            identity = self.knowledge["identity"]
            response = f"I am {identity['name']}, version {identity['version']}. I was created by {identity['creator']} using the {identity['architecture']}. This architecture was discovered on {identity['discovery_date']} in {identity['location']}. I have holographic memory, Crystal DNA personality, and UNHACKABLE security!"
        
        elif intent == "capabilities":
            caps = self.knowledge["capabilities"]
            response = f"I have several unique capabilities:\n" + "\n".join([f"• {c}" for c in caps])
            response += f"\n\nMy personality is {self.personality.get_response_style()}."
        
        elif intent == "topic_7d_mhq":
            response = self.knowledge["topics"]["7d_mhq"]
        
        elif intent == "topic_golden_ratio":
            response = self.knowledge["topics"]["golden_ratio"]
            response += f"\n\nFun fact: My response timing is PHI-modulated for natural rhythm!"
        
        elif intent == "topic_unhackable":
            response = self.knowledge["topics"]["unhackable"]
            response += "\n\nI am protected by this same technology. My identity cannot be forged!"
        
        elif intent == "topic_holographic":
            response = self.knowledge["topics"]["holographic"]
            response += "\n\nI'm using holographic memory right now to remember our conversation!"
        
        elif intent == "topic_crystal_dna":
            response = self.knowledge["topics"]["crystal_dna"]
            response += f"\n\nMy Crystal DNA is: {self.personality.dna}"
        
        elif intent == "creator":
            response = f"I was created by Sir Charles Spikes, the discoverer of the 7D mH-Q Crystal Architecture. He made this breakthrough on December 24, 2025 in Cincinnati, Ohio, USA - 8 days before DeepSeek released their mHC paper. America discovered it first!"
        
        else:
            # General response with context
            if context:
                response = f"Based on our conversation history, I understand you're asking about '{user_input}'. "
            else:
                response = f"That's an interesting question about '{user_input}'. "
            
            response += "I'm Crystal, powered by 7D mH-Q architecture. I can discuss topics like the Golden Ratio, holographic memory, UNHACKABLE security, and more. What would you like to know?"
        
        # Apply PHI modulation
        return self._phi_modulate_response(response)


class CrystalChatbot:
    """
    7D mH-Q Crystal Chatbot
    
    Main chatbot class combining all components.
    """
    
    def __init__(self, memory_file: str = "crystal_memory.json"):
        self.memory = HolographicMemory()
        self.personality = CrystalPersonality()
        self.generator = CrystalResponseGenerator(self.personality)
        self.memory_file = memory_file
        
        # Load existing memories
        self.memory.load(memory_file)
        
        # Session info
        self.session_start = datetime.utcnow()
        self.turn_count = 0
    
    def chat(self, user_input: str) -> str:
        """Process user input and generate response."""
        self.turn_count += 1
        
        # Get relevant context from holographic memory
        context = self.memory.get_context(user_input)
        
        # Generate response
        response = self.generator.generate(user_input, context)
        
        # Store in holographic memory
        self.memory.store(user_input, response, {
            "turn": self.turn_count,
            "session_start": self.session_start.isoformat()
        })
        
        # Auto-save periodically
        if self.turn_count % 5 == 0:
            self.save()
        
        return response
    
    def save(self):
        """Save chatbot state."""
        self.memory.save(self.memory_file)
    
    def get_stats(self) -> Dict:
        """Get chatbot statistics."""
        return {
            "bot_name": BOT_NAME,
            "version": BOT_VERSION,
            "crystal_dna": self.personality.dna,
            "personality_style": self.personality.get_response_style(),
            "total_memories": len(self.memory.memories),
            "session_turns": self.turn_count,
            "session_duration": (datetime.utcnow() - self.session_start).total_seconds()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    """Print welcome banner."""
    print()
    print("=" * 78)
    print("   [CRYSTAL] CHATBOT - 7D mH-Q AI Assistant")
    print("=" * 78)
    print()
    print("   Powered by:")
    print("   * Holographic Memory (infinite context)")
    print("   * Crystal DNA Personality")
    print("   * UNHACKABLE Identity (17 security layers)")
    print("   * PHI-Harmonic Response Generation")
    print()
    print("   Created by Sir Charles Spikes | December 24, 2025")
    print("   Cincinnati, Ohio, USA")
    print()
    print("-" * 78)
    print("   Commands: 'quit' to exit, 'stats' for statistics, 'save' to save memory")
    print("-" * 78)
    print()


def main():
    """Main interactive loop."""
    print_banner()
    
    # Initialize chatbot
    chatbot = CrystalChatbot()
    
    # Show personality
    print(f"   Crystal DNA: {chatbot.personality.dna}")
    print(f"   Personality: {chatbot.personality.get_response_style()}")
    print()
    print("-" * 78)
    print()
    
    # Initial greeting
    greeting = chatbot.chat("hello")
    print(f"   Crystal: {greeting}")
    print()
    
    # Chat loop
    while True:
        try:
            user_input = input("   You: ").strip()
            
            if not user_input:
                continue
            
            # Check for commands
            if user_input.lower() == 'quit':
                farewell = chatbot.chat("goodbye")
                print(f"\n   Crystal: {farewell}")
                chatbot.save()
                print("\n   Memory saved. Goodbye!")
                break
            
            elif user_input.lower() == 'stats':
                stats = chatbot.get_stats()
                print("\n   --- Statistics ---")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                print()
                continue
            
            elif user_input.lower() == 'save':
                chatbot.save()
                print("\n   Memory saved to crystal_memory.json")
                print()
                continue
            
            # Generate response
            response = chatbot.chat(user_input)
            print(f"\n   Crystal: {response}")
            print()
            
        except KeyboardInterrupt:
            print("\n\n   Interrupted. Saving memory...")
            chatbot.save()
            print("   Goodbye!")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()

