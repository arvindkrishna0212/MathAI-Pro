from typing import List, Dict, Any
import json
import aiohttp
from ..models.schemas import WebSearchResult
from ..config import settings
from tavily import TavilyClient

class WebSearchService:
    def __init__(self):
        self.tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        self.mcp_endpoint = settings.MCP_ENDPOINT
        self.mcp_api_key = settings.MCP_API_KEY

    async def search(self, query: str, max_results: int = 3) -> List[WebSearchResult]:
        print(f"Performing web search for: {query}")
        
        # First try MCP if configured
        if self.mcp_endpoint and self.mcp_api_key:
            try:
                mcp_results = await self._search_with_mcp(query, max_results)
                if mcp_results:
                    return mcp_results
            except Exception as e:
                print(f"Error during MCP search, falling back to Tavily: {e}")
        
        # Fallback to Tavily
        try:
            response = self.tavily_client.search(query=query, max_results=max_results, include_raw_html=False, include_answer=False)
            
            results = []
            for r in response['results']:
                results.append(WebSearchResult(
                    title=r['title'],
                    url=r['url'],
                    snippet=r['content']
                ))
            return results
        except Exception as e:
            print(f"Error during Tavily web search: {e}")
            return []
    
    async def _search_with_mcp(self, query: str, max_results: int = 3) -> List[WebSearchResult]:
        """Perform search using Model Context Protocol (MCP)"""
        print(f"Performing MCP search for: {query}")
        
        # Construct MCP request
        mcp_request = {
            "messages": [
                {"role": "user", "content": f"Search for information about: {query}"},
            ],
            "context": {
                "documents": [],
                "tools": [{
                    "type": "search",
                    "params": {
                        "query": query,
                        "max_results": max_results
                    }
                }]
            }
        }
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.mcp_api_key}"
            }
            
            async with session.post(self.mcp_endpoint, headers=headers, json=mcp_request) as response:
                if response.status != 200:
                    print(f"MCP search failed with status {response.status}")
                    return []
                
                data = await response.json()
                
                # Extract search results from MCP response
                results = []
                if "context" in data and "documents" in data["context"]:
                    for doc in data["context"]["documents"]:
                        if "source" in doc and "title" in doc and "content" in doc:
                            results.append(WebSearchResult(
                                title=doc["title"],
                                url=doc["source"],
                                snippet=doc["content"]
                            ))
                return results

