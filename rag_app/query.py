import openai
from rag_app.models import TextChunk
from lancedb import connect
from typing import List
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box
import duckdb
import typer

app = typer.Typer()


@app.command(help="Query LanceDB for some results")
def db(
    db_path: str = typer.Option(help="Your LanceDB path"),
    table_name: str = typer.Option(help="Table to ingest data into"),
    query: str = typer.Option(help="Text to query against existing vector db chunks"),
    n: int = typer.Option(default=3, help="Maximum number of chunks to return"),
):
    """
    Query LanceDB for some results.

    Args:
        db_path (str): Your LanceDB path.
        table_name (str): Table to ingest data into.
        query (str): Text to query against existing vector db chunks.
        n (int): Maximum number of chunks to return.

    Raises:
        ValueError: If the database path does not exist.

    Returns:
        None
    """
    if not Path(db_path).exists():
        raise ValueError(f"Database path {db_path} does not exist.")
    db = connect(db_path)
    db_table = db.open_table(table_name)

    client = openai.OpenAI()
    query_vector = (
        client.embeddings.create(
            input=query, model="text-embedding-3-large", dimensions=256
        )
        .data[0]
        .embedding
    )

    results: List[TextChunk] = (
        db_table.search(query_vector).limit(n).to_pydantic(TextChunk)
    )

    sql_table = db_table.to_lance()
    df = duckdb.query(
        "SELECT doc_id, count(chunk_id) as count FROM sql_table GROUP BY doc_id"
    ).to_df()

    doc_id_to_count = df.set_index("doc_id")["count"].to_dict()

    table = Table(title="Results", box=box.HEAVY, padding=(1, 2), show_lines=True)
    table.add_column("Chunk Id", style="magenta")
    table.add_column("Content", style="magenta", max_width=120)
    table.add_column("Post Title", style="green", max_width=30)
    table.add_column("Chunk Number", style="yellow")
    table.add_column("Publish Date", style="blue")

    for result in results:
        chunk_number = f"{result.chunk_number}/{doc_id_to_count[result.doc_id]}"
        table.add_row(
            result.chunk_id,
            f"{result.post_title}({result.source})",
            result.text,
            chunk_number,
            result.publish_date.strftime("%Y-%m"),
        )
    Console().print(table)
