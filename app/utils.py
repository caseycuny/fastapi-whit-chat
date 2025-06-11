import aiohttp
import logging
from typing import Dict, Any

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
