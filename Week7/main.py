"""
Command Line Interface for the RAG Document Question Answering System

Usage
-----

Build Index
    python main.py ingest

Ask Question
    python main.py query "What is Retrieval-Augmented Generation?"
"""

import sys
from src.rag_pipeline import RAGPipeline


def print_banner():
    print("\n" + "=" * 70)
    print("      DOCUMENT QUESTION ANSWERING SYSTEM (RAG)")
    print("=" * 70)


def print_help():
    print("""
Usage:

Build Knowledge Base
--------------------
python main.py ingest

Ask a Question
--------------
python main.py query "Your question here"

Examples
--------
python main.py query "What is the main topic?"
python main.py query "Summarize this document."
""")


def main():

    print_banner()

    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].lower()

    pipeline = RAGPipeline()

    try:

        # ==================================================
        # INGEST
        # ==================================================

        if command == "ingest":

            pipeline.ingest()

            print("\n✓ Knowledge base built successfully.")

        # ==================================================
        # QUERY
        # ==================================================

        elif command == "query":

            if len(sys.argv) < 3:
                print("❌ Please provide a question.")
                print_help()
                return

            question = " ".join(sys.argv[2:])

            result = pipeline.query(question)

            print("\n" + "=" * 70)

            print("QUESTION")
            print("-" * 70)
            print(result["question"])

            print("\nANSWER")
            print("-" * 70)
            print(result["answer"])

            print("\nRETRIEVED SOURCES")
            print("-" * 70)

            for source in result["sources"]:

                print(
                    f"[Rank {source.get('rank', '-')}] "
                    f"{source['source']} | "
                    f"Chunk {source['chunk_id']} | "
                    f"Score {source['score']:.4f}"
                )

            metrics = result.get("metrics", {})

            if metrics:

                print("\nSYSTEM METRICS")
                print("-" * 70)

                print(f"Embedding Time : {metrics['embedding_time']} sec")
                print(f"Retrieval Time : {metrics['retrieval_time']} sec")
                print(f"Generation Time: {metrics['generation_time']} sec")
                print(f"Total Time     : {metrics['total_time']} sec")

            print("=" * 70)

        else:

            print(f"❌ Unknown command: {command}")

            print_help()

    except Exception as e:

        print("\n❌ ERROR")
        print("-" * 70)
        print(e)


if __name__ == "__main__":
    main()