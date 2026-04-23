"""
Tool 1: arXiv paper search tool
Function: Search for academic papers by keyword, returning titles, authors, abstracts, and links
"""

import arxiv


def search_arxiv(query: str, max_results: int = 5) -> list:
    """
    Search arXiv for papers

    Args:
        query: Search keywords (English works best)
        max_results: Maximum number of papers to return, default 5

    Returns:
        List of papers, each containing title/authors/abstract/link/publication date
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance  # Sort by relevance
    )

    results = []
    for paper in client.results(search):
        results.append({
            "title": paper.title,
            "authors": [str(a) for a in paper.authors[:3]],  # Only take the first 3 authors
            "abstract": paper.summary[:500] + "..." if len(paper.summary) > 500 else paper.summary,
            "url": paper.pdf_url,
            "published": str(paper.published.date()),
            "categories": paper.categories[:3]
        })
    return results


def format_papers(papers: list) -> str:
    """Format a list of papers into a readable string"""
    if not papers:
        return "No papers found. Try different keywords."

    output = f"Found {len(papers)} papers:\n\n"
    for i, paper in enumerate(papers, 1):
        authors_str = ", ".join(paper["authors"])
        if len(paper["authors"]) == 3:
            authors_str += " et al."
        output += (
            f"Paper {i}: {paper['title']}\n"
            f"Authors: {authors_str}\n"
            f"Published: {paper['published']}\n"
            f"Categories: {', '.join(paper['categories'])}\n"
            f"Abstract: {paper['abstract']}\n"
            f"PDF: {paper['url']}\n\n"
        )
    return output
