import os
import sys

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

from keys import OPENAI_API_KEY

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    query = sys.argv[1]

    loader = TextLoader("data.txt")
    index = VectorstoreIndexCreator().from_loaders([loader])

    print(index.query(query, llm=ChatOpenAI()))
