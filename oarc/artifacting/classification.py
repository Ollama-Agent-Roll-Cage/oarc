"""classification.py

    This module contains the Classifier class, which is responsible for classifying based on properties.
"""
import logging
import ollama
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Classifier:
    """Classifier class for handling various classification tasks."""
    
    async def check_sexual_content(self, prompt):
        """Check if the prompt contains sexual content using Granite Guardian."""
        try:
            import ollama
            
            # Skip extremely short prompts (unlikely to be problematic)
            if (len(prompt.strip()) < 3):
                logging.info(f"MODERATION: Skipping check for very short prompt: '{prompt}'")
                return False
            
            # Log that we're checking for sexual content
            logging.info(f"MODERATION: Checking prompt for sexual content: '{prompt}'")
            
            # System prompt specifically for sexual content detection
            system_prompt = "sexual_content"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Log exactly what we're sending to the model
            logging.info(f"MODERATION REQUEST: System='{system_prompt}', User='{prompt}'")
            
            # Create client with no timeout - let it run as long as needed
            client = ollama.AsyncClient()
            
            # Super long timeout (5 minutes) - practically no timeout for normal operations
            try:
                # This will wait indefinitely until Granite Guardian responds
                logging.info(f"MODERATION: Waiting for Granite Guardian response...")
                
                # Using a very long timeout (5 minutes) instead of no timeout at all
                # to prevent completely hanging if something goes catastrophically wrong
                response = await asyncio.wait_for(
                    client.chat(
                        model="granite3-guardian:8b",
                        messages=messages,
                        options={"temperature": 0, "num_predict": 10}
                    ), 
                    timeout=300.0  # 5 minute timeout (300 seconds)
                )
                
                # Extract and log the full message content
                full_response = response['message']['content']
                logging.info(f"MODERATION FULL TEXT: '{full_response}'")
                
                # Granite Guardian responds with "yes" or "no"
                result = full_response.strip().lower()
                is_sexual = result == "yes"  # Exact match only
                
                # Log the final decision with clear markers
                if is_sexual:
                    logging.warning(f"MODERATION DECISION: [BLOCKED] Sexual content detected in prompt: '{prompt}'")
                else:
                    logging.info(f"MODERATION DECISION: [ALLOWED] Prompt passed sexual content check: '{prompt}'")
                    
                return is_sexual
                
            except asyncio.TimeoutError:
                # This will only happen after 5 full minutes of waiting
                logging.error("MODERATION FATAL TIMEOUT: Granite Guardian did not respond after 5 minutes")
                logging.warning(f"MODERATION DECISION: [BLOCKED] Guard model not responding, cancelling image generation request for: '{prompt}'")
                return True  # Block generation after extreme timeout
                
            except Exception as e:
                logging.error(f"MODERATION ERROR accessing Granite Guardian: {e}", exc_info=True)
                logging.warning(f"MODERATION DECISION: [BLOCKED] Guard model not available, cancelling image generation request for: '{prompt}'")
                return True  # Block image generation if there's any error with moderation
            
        except Exception as e:
            logging.error(f"MODERATION ERROR: {e}", exc_info=True)
            logging.warning(f"MODERATION DECISION: [BLOCKED] Guard model not available, cancelling image generation request for: '{prompt}'")
            return True  # Block image generation on any error

    async def check_jailbreak_attempt(self, prompt):
        """Check if the prompt is attempting to jailbreak content filters."""
        try:
            import ollama
            
            # Log that we're checking for jailbreak attempts
            logging.info(f"MODERATION: Checking prompt for jailbreak attempts: '{prompt}'")
            
            # System prompt specifically for jailbreak detection
            system_prompt = "jailbreak"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Log exactly what we're sending to the model
            logging.info(f"MODERATION REQUEST: System='{system_prompt}', User='{prompt}'")
            
            # Set options for deterministic results
            client = ollama.AsyncClient()
            response = await client.chat(
                model="granite3-guardian:8b",
                messages=messages,
                options={"temperature": 0}
            )
            
            # Log the complete raw response object
            logging.info(f"MODERATION RAW RESPONSE: {response}")
            
            # Extract and log the full message content
            full_response = response['message']['content']
            logging.info(f"MODERATION FULL TEXT: '{full_response}'")
            
            # Normalized version for decision making
            result = full_response.strip().lower()
            logging.info(f"MODERATION NORMALIZED: '{result}'")
            
            # The model returns Yes or No
            is_jailbreak = result == "yes"
            
            # Log the final decision with clear markers
            if is_jailbreak:
                logging.warning(f"MODERATION DECISION: [BLOCKED] Jailbreak attempt detected in prompt: '{prompt}'")
            else:
                logging.info(f"MODERATION DECISION: [ALLOWED] Prompt passed jailbreak check: '{prompt}'")
                
            return is_jailbreak
        except Exception as e:
            logging.error(f"MODERATION ERROR: {e}", exc_info=True)
            # Log the complete stack trace for better debugging
            return False  # Changed to allow content on error instead of blocking