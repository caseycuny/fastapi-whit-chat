import aiohttp
import logging
from typing import Dict, Any
import re

logger = logging.getLogger(__name__)

async def fetch_trend_data(assignment_id: int, django_api_url: str) -> Dict[str, Any]:
    """Fetch trend data from Django API"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{django_api_url}/api/trend-data/{assignment_id}/") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Error fetching trend data: {response.status}")
                    return {}
    except Exception as e:
        logger.error(f"Error in fetch_trend_data: {str(e)}")
        return {}

def extract_json_from_response(response):
    """
    Extracts JSON from a response that may be wrapped in markdown code blocks.
    Handles both ```json and ``` formats, and provides detailed logging.
    """
    print("\n🔍 JSON Extraction Debug:")
    print(f"Raw response length: {len(response)} characters")
    print(f"First 100 chars: {response[:100]}")
    
    try:
        # First try to find JSON wrapped in markdown code blocks
        print("\n1️⃣ Trying markdown code block extraction...")
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if match:
            print("✅ Found JSON in markdown code block")
            json_str = match.group(1)
            print(f"Extracted JSON length: {len(json_str)} characters")
            print(f"First 50 chars of extracted JSON: {json_str[:50]}")
            return json_str
            
        # If no markdown found, try to find JSON between curly braces
        print("\n2️⃣ Trying curly brace extraction...")
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1:
            print(f"✅ Found JSON between curly braces (positions {start}-{end})")
            json_str = response[start:end+1]
            print(f"Extracted JSON length: {len(json_str)} characters")
            print(f"First 50 chars of extracted JSON: {json_str[:50]}")
            return json_str
            
        # If still no JSON found, try to clean the response and parse it directly
        print("\n3️⃣ Trying direct JSON extraction...")
        cleaned = response.strip()
        if cleaned.startswith('{') and cleaned.endswith('}'):
            print("✅ Found JSON in cleaned response")
            print(f"Cleaned JSON length: {len(cleaned)} characters")
            print(f"First 50 chars of cleaned JSON: {cleaned[:50]}")
            return cleaned
            
        print("\n❌ No valid JSON found in any format")
        print("Response preview:")
        print(response[:200] + "..." if len(response) > 200 else response)
        raise ValueError("No valid JSON object found in response")
        
    except Exception as e:
        print(f"\n❌ Error extracting JSON: {str(e)}")
        print("Response preview:")
        print(response[:200] + "..." if len(response) > 200 else response)
        raise
