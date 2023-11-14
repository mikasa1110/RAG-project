from langchain.document_loaders import NotionDirectoryLoader
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# import notion loader and load md file 
loader = NotionDirectoryLoader("notion_db")
pages = loader.load()
md_file = pages[0].page_content

from langchain.text_splitter import MarkdownHeaderTextSplitter
# set headers 
headers_to_split_on = [
    ("#", "Title"),
    ("##", "Section"),
    ("###", "Details"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(md_file)
# split recursively
from langchain.text_splitter import RecursiveCharacterTextSplitter

chunk_size = 500
chunk_overlap = 0
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)
all_splits = text_splitter.split_documents(md_header_splits)
# Build vectorstore and keep the metadata
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=all_splits, embedding=HuggingFaceEmbeddings())

# Loading llm and create metadata infos
from langchain.llms import LlamaCpp
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="Title",
        description="the first header in the document,introduction",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="Section",
        description="the second header in the document,middle part",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="Details",
        description="the third header in the document,about the details",
        type="string or list[string]",
    ),
]
document_content_description = "the content of the document,related with fruit and vegetable"

# Define self query retriever
n_gpu_layers = 10  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/data2/home/wtzhang/llama-RAG-test/models/ggml-model-q4_0.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
qa_chain.run("Summarize the Detail part 'Spinach' of the document")