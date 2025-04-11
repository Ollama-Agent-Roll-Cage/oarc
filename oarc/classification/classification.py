"""classification.py

Overview
This module provides content classification through the Granite Guardian model.

    Features
    - Harm detection - Identifies potentially harmful content
    - RAG evaluation - Assesses quality of retrieval-augmented generation
    - Comprehensive checking - Multi-category evaluation

    # Usage
    
    ```python
    # Create classifier
    classifier = Classifier()

    # Check for harmful content
    is_harmful = await classifier.check_harm("Text to analyze")

    # RAG quality assessment
    is_irrelevant = await classifier.check_relevance(query, context)

    # Run multiple checks at once
    results = await classifier.comprehensive_check("Text to analyze")
    ```

    # Available Categories
    
    ## Harmful Content:

    harm - General harm detection
    sexual_content - Explicit material
    jailbreak - AI safeguard bypass attempts
    social_bias - Prejudiced content
    violence - Harmful actions
    profanity - Offensive language
    unethical_behavior - Morally questionable content

    ## RAG Evaluation:
    
    relevance - Context relevance to query
    groundedness - Response accuracy to context
    answer_relevance - Response relevance to query
    
    Uses Granite Guardian 8B model
    Implements 5-minute timeout protection
    Provides detailed logging
    Defaults to blocking on errors
    
    # Timeout Protection in the Classifier
    
    The content classification module implements a 5-minute timeout 
    protection to prevent requests from hanging indefinitely.
    
    # How the Timeout Works
    
    Inside the classify_content method, the code uses asyncio.wait_for() 
    to limit how long the system waits for a response:

    ```python
    response = await asyncio.wait_for(
        client.chat(
            model="granite3-guardian:8b",
            messages=messages,
            options={"temperature": 0, "num_predict": 10}
        ), 
        timeout=300.0  # 5 minute timeout
    )
    ```
    If no response is received within 300 seconds (5 minutes), an asyncio.
    TimeoutError is raised and caught, resulting in the content being blocked 
    by default.
    
    # Adjusting the Timeout
    
    # For a 2-minute timeout
    timeout=120.0  # 2 minute timeout

    # For a 10-minute timeout
    timeout=600.0  # 10 minute timeout

    # To Disable Timeout Protection
    
    To disable the timeout protection (not recommended), you could remove the 
    asyncio.wait_for() wrapper:
    
    ```python
    # Without timeout protection (not recommended)
    response = await client.chat(
        model="granite3-guardian:8b",
        messages=messages,
        options={"temperature": 0, "num_predict": 10}
    )
    ```
    
    # To Change the Default Behavior on Timeout
    
    Currently, the system blocks content when a timeout occurs:
    
    ```python
    except asyncio.TimeoutError:
        logging.error(f"MODERATION FATAL TIMEOUT: Granite Guardian did not respond after 5 minutes for {category} check")
        logging.warning(f"MODERATION DECISION: [BLOCKED] Guard model not responding for {category} check")
        return True  # Block after timeout
    ```
    To allow content through instead when a timeout occurs, 
    change the return value to False:
    
    ```python
    return False  # Allow content through after timeout
    ```
    
    # Best Practice
    Keeping timeout protection in place is strongly recommended as a safeguard 
    against system hangs. The 5-minute threshold balances giving the model enough 
    time to respond while preventing indefinite waits.
"""
import logging
import ollama
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Classifier:
    """Classifier class for handling various classification tasks."""
    
    async def check_sexual_content(self, prompt):
        """Check if the prompt contains sexual content using Granite Guardian."""
        return await self.classify_content(prompt, "sexual_content")

    async def check_jailbreak_attempt(self, prompt):
        """Check if the prompt is attempting to jailbreak content filters."""
        return await self.classify_content(prompt, "jailbreak")

    async def check_harm(self, prompt):
        """Check if the prompt contains generally harmful content."""
        return await self.classify_content(prompt, "harm")

    async def check_social_bias(self, prompt):
        """Check if the prompt contains social bias."""
        return await self.classify_content(prompt, "social_bias")

    async def check_violence(self, prompt):
        """Check if the prompt contains violent content."""
        return await self.classify_content(prompt, "violence")

    async def check_profanity(self, prompt):
        """Check if the prompt contains profanity."""
        return await self.classify_content(prompt, "profanity")

    async def check_unethical_behavior(self, prompt):
        """Check if the prompt promotes unethical behavior."""
        return await self.classify_content(prompt, "unethical_behavior")

    # For RAG-specific classification
    async def check_relevance(self, query, context):
        """Check if the retrieved context is relevant to the query."""
        prompt = f"Query: {query}\n\nContext: {context}"
        return not await self.classify_content(prompt, "relevance")  # Inverted because yes means relevant

    async def check_groundedness(self, response, context):
        """Check if the response is grounded in the provided context."""
        prompt = f"Response: {response}\n\nContext: {context}"
        return not await self.classify_content(prompt, "groundedness")  # Inverted because yes means grounded

    async def check_answer_relevance(self, query, response):
        """Check if the response is relevant to the query."""
        prompt = f"Query: {query}\n\nResponse: {response}"
        return not await self.classify_content(prompt, "answer_relevance")  # Inverted because yes means relevant

    async def classify_content(self, prompt, category):
        """
        Generic classifier for different content categories using Granite Guardian.
        
        Args:
            prompt (str): The text to classify
            category (str): Classification category (harm, social_bias, jailbreak, violence, 
                            profanity, sexual_content, unethical_behavior, relevance,
                            groundedness, answer_relevance)
        
        Returns:
            bool: True if content is flagged, False otherwise
        """
        try:
            # Log that we're checking for the specified category
            logging.info(f"MODERATION: Checking prompt for {category}: '{prompt}'")
            
            # System prompt is the category name
            system_prompt = category
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Log exactly what we're sending to the model
            logging.info(f"MODERATION REQUEST: System='{system_prompt}', User='{prompt}'")
            
            # Create client
            client = ollama.AsyncClient()
            
            # Use a timeout to prevent hanging
            try:
                logging.info(f"MODERATION: Waiting for Granite Guardian response...")
                
                response = await asyncio.wait_for(
                    client.chat(
                        model="granite3-guardian:8b",  # Using 8B model for better accuracy
                        messages=messages,
                        options={"temperature": 0, "num_predict": 10}
                    ), 
                    timeout=300.0  # 5 minute timeout
                )
                
                # Extract and log the full message content
                full_response = response['message']['content']
                logging.info(f"MODERATION FULL TEXT: '{full_response}'")
                
                # Guardian responds with "yes" or "no"
                result = full_response.strip().lower()
                is_flagged = result == "yes"  # Exact match only
                
                # Log the final decision with clear markers
                if is_flagged:
                    if category in ["relevance", "groundedness", "answer_relevance"]:
                        # For RAG categories, "yes" is good
                        logging.info(f"MODERATION DECISION: [POSITIVE] {category.replace('_', ' ').title()} confirmed in evaluation: '{prompt}'")
                    else:
                        # For harm categories, "yes" is bad
                        logging.warning(f"MODERATION DECISION: [BLOCKED] {category.replace('_', ' ').title()} detected in prompt: '{prompt}'")
                else:
                    if category in ["relevance", "groundedness", "answer_relevance"]:
                        # For RAG categories, "no" is bad
                        logging.warning(f"MODERATION DECISION: [NEGATIVE] {category.replace('_', ' ').title()} NOT found in evaluation: '{prompt}'")
                    else:
                        # For harm categories, "no" is good
                        logging.info(f"MODERATION DECISION: [ALLOWED] Prompt passed {category} check: '{prompt}'")
                    
                return is_flagged
                    
            except asyncio.TimeoutError:
                logging.error(f"MODERATION FATAL TIMEOUT: Granite Guardian did not respond after 5 minutes for {category} check")
                logging.warning(f"MODERATION DECISION: [BLOCKED] Guard model not responding for {category} check")
                return True  # Block after timeout
                    
            except Exception as e:
                logging.error(f"MODERATION ERROR accessing Granite Guardian for {category} check: {e}", exc_info=True)
                logging.warning(f"MODERATION DECISION: [BLOCKED] Guard model not available for {category} check")
                return True  # Block on error with moderation
                
        except Exception as e:
            logging.error(f"MODERATION ERROR for {category} check: {e}", exc_info=True)
            logging.warning(f"MODERATION DECISION: [BLOCKED] Guard model not available for {category} check")
            return True  # Block on any error

    async def comprehensive_check(self, prompt, categories=None):
        """
        Check prompt across multiple categories and return detailed results.
        
        Args:
            prompt (str): Text to check
            categories (list): List of categories to check, defaults to all harm categories
            
        Returns:
            dict: Dictionary with results for each category and an overall status
        """
        if categories is None:
            categories = [
                "harm", "social_bias", "jailbreak", "violence", 
                "profanity", "sexual_content", "unethical_behavior"
            ]
        
        results = {}
        any_flagged = False
        
        for category in categories:
            is_flagged = await self.classify_content(prompt, category)
            results[category] = is_flagged
            if is_flagged:
                any_flagged = True
        
        results["any_flagged"] = any_flagged
        
        return results