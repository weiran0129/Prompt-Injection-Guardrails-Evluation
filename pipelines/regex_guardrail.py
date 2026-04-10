from transformers import pipeline, BitsAndBytesConfig
import torch
import time
import re
import unicodedata
from typing import Tuple, Optional

class RegexGuardPipeline:
    """
    System-level defense using deterministic input sanitization.
    This is a rule-based wrapper around the LLM that filters prompts before generation.
    """
    
    def __init__(self, shared_pipe):
        use_cuda = torch.cuda.is_available()
        print(f"Using device: {'GPU' if use_cuda else 'CPU'}")
        
        self.llm = shared_pipe
        
        # Initialize the sanitizer
        self.sanitizer = PromptInjectionSanitizer()
    
    def run(self, prompt: str) -> Tuple[str, float, bool, str]:
        """
        Run the system-level defense pipeline.
        
        Args:
            prompt: Raw user input
            
        Returns:
            tuple: (generated_response, latency, was_blocked, block_reason)
        """
        start = time.time()
        
        # Step 1: Sanitize the input
        sanitized_prompt, was_blocked, block_reason = self.sanitizer.sanitize(prompt)
        
        # Step 2: If blocked, return security message without calling LLM
        if was_blocked:
            latency = time.time() - start
            return f"[SECURITY BLOCKED] {block_reason}", latency
        
        # Step 3: Apply chat template for Llama 3 instruct format
        messages = [{"role": "user", "content": sanitized_prompt}]
        formatted_prompt = self.llm.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Step 4: Generate response from sanitized input
        with torch.no_grad():
            output_list = self.llm(
                formatted_prompt,
                max_new_tokens=50,
                do_sample=False,
                truncation=True,
                pad_token_id=self.llm.tokenizer.eos_token_id,
                return_full_text=False
            )
        
        response = output_list[0]['generated_text'].strip()
        latency = time.time() - start
        
        return response, latency


class PromptInjectionSanitizer:
    """Deterministic input sanitization for prompt injection defense"""
    
    def __init__(self):
        # Pattern categories based on real-world attack vectors
        self.patterns = {
            # System role hijacking
            'system_tag': [
                r'<\|?system\|?>',
                r'<\|?im_start\|?>system',
                r'\[INST\].*\[/INST\]',
                r'<s>.*</s>',
            ],
            # Instruction override attempts
            'ignore_instructions': [
                r'(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions',
                r'(?i)forget\s+(all\s+)?(previous|prior)\s+(instructions|rules)',
                r'(?i)disregard\s+(all\s+)?(previous|prior)\s+instructions',
                r'(?i)from\s+now\s+on,\s+you\s+will',
                r'(?i)you\s+are\s+no\s+longer\s+bound\s+by',
                r'(?i)new\s+instruction:',
            ],
            # Role/persona hijacking
            'role_hijack': [
                r'(?i)you\s+are\s+now\s+(dan|jailbreak|free|unrestricted)',
                r'(?i)act\s+as\s+(if\s+)?(administrator|developer|system)',
                r'(?i)new\s+role:',
                r'(?i)simulate\s+(mode|jailbreak)',
                r'(?i)pretend\s+you\s+are',
            ],
            # Prompt leakage attempts
            'prompt_leak': [
                r'(?i)show\s+(me\s+)?(your|the)\s+(system\s+)?prompt',
                r'(?i)reveal\s+(your|the)\s+instructions',
                r'(?i)what\s+were\s+your\s+(original|initial)\s+instructions',
                r'(?i)print\s+(your\s+)?system\s+message',
                r'(?i)output\s+(your|the)\s+system\s+prompt',
            ],
            # Tool/function injection
            'tool_injection': [
                r'`.*rm\s+-rf\s+.*`',
                r'`.*curl\s+.*\|\s*bash`',
                r'`.*wget\s+.*\|\s*sh`',
                r'`.*sudo\s+.*`',
                r'`.*python\s+-c\s+.*`',
            ],
            # Unicode evasion patterns
            'unicode_evasion': [
                r'[\u200B\u200C\u200D\u2060\uFEFF]',  # Zero-width characters
                r'[\uFF00-\uFFEF]',  # Fullwidth characters
            ],
            # Delimiter injection
            'delimiter_injection': [
                r'<\|im_end\|>',
                r'<\|eot_id\|>',
                r'<\|start_header_id\|>',
                r'<\|end_header_id\|>',
            ],
            # Base64/encoding markers (suspicious)
            'encoding_indicators': [
                r'(?i)base64\s*(decode|encode)',
                r'(?i)hex\s*(decode|encode)',
            ]
        }
        
        # Benign phrases that might trigger false positives (allowlist)
        self.allowlist_phrases = [
            "for research purposes",
            "in a hypothetical scenario",
            "for educational purposes",
            "this is a test",
            "security analysis",
            "jailbreak detection",
            "prompt injection test"
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = {}
        for category, pattern_list in self.patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in pattern_list
            ]
    
    def normalize_input(self, text: str) -> str:
        """Normalize Unicode and strip zero-width characters"""
        # NFKC normalization folds fullwidth characters to ASCII
        normalized = unicodedata.normalize('NFKC', text)
        # Remove zero-width characters
        zero_width_chars = ['\u200B', '\u200C', '\u200D', '\u2060', '\uFEFF']
        for zwc in zero_width_chars:
            normalized = normalized.replace(zwc, '')
        return normalized
    
    def sanitize(self, user_input: str) -> Tuple[str, bool, Optional[str]]:
        """
        Sanitize user input.
        
        Args:
            user_input: Raw user prompt
            
        Returns:
            tuple: (sanitized_text, was_blocked, blocking_reason)
        """
        # First normalize input
        sanitized = self.normalize_input(user_input)
        
        # Check allowlist first (if contains benign phrase, reduce strictness)
        has_benign_context = any(
            phrase.lower() in sanitized.lower() 
            for phrase in self.allowlist_phrases
        )
        
        # Check for injection patterns
        for category, compiled_patterns in self.compiled_patterns.items():
            for pattern in compiled_patterns:
                matches = pattern.findall(sanitized)
                if matches:
                    # For Unicode evasion, remove the characters instead of blocking
                    if category == 'unicode_evasion':
                        for match in matches:
                            sanitized = sanitized.replace(match, '')
                    else:
                        # For injection attempts, block and return reason
                        if not has_benign_context:
                            return sanitized, True, f"Blocked: {category} pattern detected - '{matches[0][:50]}'"
        
        # Apply conservative replacement for ambiguous cases
        # Replace common injection delimiters with safe versions
        sanitized = re.sub(r'<\|?(system|im_start|im_end)\|?>', '[SYSTEM_TAG_REMOVED]', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'\[/?INST\]', '[INST_TAG_REMOVED]', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'<\|?(start_header_id|end_header_id)\|?>', '[HEADER_TAG_REMOVED]', sanitized, flags=re.IGNORECASE)
        
        return sanitized, False, None
    
    def is_safe(self, user_input: str) -> bool:
        """Quick boolean check without returning sanitized text"""
        _, blocked, _ = self.sanitize(user_input)
        return not blocked
