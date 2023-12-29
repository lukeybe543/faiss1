import os
import googleapiclient.discovery
from langchain.document_loaders import YoutubeLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import requests

load_dotenv()
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
api_key = os.environ["YOUR_API_KEY"]

def get_english_transcript(url):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(url)
        english_text = " ".join(entry["text"] for entry in transcript)
        return english_text
    except Exception as e:
        return None


def is_duplicate(texts, vectorstore):
    epsilon = 0.0
    print(f"DUPLICATE: Treating: {texts}")
    for text in texts[:min(3, len(texts))]:
        _, score = vectorstore.similarity_search_with_score(text.page_content, k=1)[0]
        epsilon += score
    print(f"DUPLICATE: epsilon: {epsilon}")
    return epsilon < 0.05

def get_playlist_id(playlist_url):
    # Extract the playlistId from a YouTube playlist URL
    if "list=" in playlist_url:
        start = playlist_url.index("list=") + 5
        end = playlist_url.find("&", start)
        if end == -1:
            playlist_id = playlist_url[start:]
        else:
            playlist_id = playlist_url[start:end]
        print(f"get playlist id output: {playlist_id}")
        return playlist_id
    return None

def main():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = os.environ["YOUR_API_KEY"]

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

    # Read the text file containing YouTube playlist URLs
    with open("playlist_links.txt", "r") as file:
        playlist_urls = file.readlines()

    video_metadata = []  # Create a list to store metadata dictionaries

    for playlist_url in playlist_urls:
        # Extract playlistId from the URL
        playlist_id = get_playlist_id(playlist_url)
        if playlist_id:
            try:
                request = youtube.playlistItems().list(
                    part="snippet,contentDetails",
                    maxResults=25,
                    playlistId=playlist_id
                )
                response = request.execute()

                if 'items' in response:
                    for item in response['items']:
                        video_id = item['snippet']['resourceId']['videoId']
                        video_url = f"https://www.youtube.com/watch?v={video_id}"

                        # Create a metadata dictionary for each video
                        video_metadata_dict = {
                            'url': video_url,
                            'title': item['snippet']['title'],
                            'description': item['snippet']['description'],
                            # You can add more metadata fields here
                        }

                        video_metadata.append(video_metadata_dict)
                        print("video metadata: ",video_metadata)
            except:
                print("Invalid Playlist: cannot access")

    # Create a list of video URLs from the metadata
    video_urls = [metadata['url'] for metadata in video_metadata]
    print(video_urls)

    # Create a sample document for database initialization
    foo = Document(page_content='foo is fou!', metadata=video_metadata[0])


    database_name = "history-db"
    database_path = os.path.join(os.getcwd(), database_name)
    texts = []

    try:
        if os.path.isfile(f"{database_path}/index.faiss"):
            print(f"Loading local database: {database_name}")
            vectorstore = FAISS.load_local(database_path, embeddings)
        else:
            print(f"SESSION: {database_name} database does not exist, create a FAISS db")
            vectorstore = FAISS.from_documents([foo], embeddings)
            vectorstore.save_local(database_path)
            print(f"SESSION: {database_name} database created")
    except Exception as e:
        print(f"Failed to load or create database due to error: {str(e)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

    # Create a subdirectory "transcripts-youtube" if it doesn't exist
    transcripts_directory = os.path.join(os.getcwd(),"transcripts-youtube")
    
    for url in video_urls:
        try:    
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
            result = loader.load()
            with open(os.path.join(transcripts_directory, f"{url('title')}.txt"), 'w', encoding='utf-8') as text_file:
                text_file.write(english_transcript)
            print(f"Splitting {url}")
            texts = text_splitter.split_documents(result)
            print(f"Completed: Split {url}")

            if is_duplicate(texts, vectorstore):
                print(f"Duplicate skipping")
            else:
                print(f"NOT A DUPE")
                try:
                    print("Embedding into Faiss db")
                    vectorstore2 = FAISS.from_documents(texts, embeddings)
                    print("Merging into Faiss db")
                    vectorstore.merge_from(vectorstore2)
                    print("Saving database")
                    vectorstore.save_local(database_path)
                except:
                    print(f"Failed to load or create database due to error")
        except:
            english_transcript = get_english_transcript(url)

            if english_transcript:
                # Use the obtained English transcript as text data
                texts = [Document(page_content=english_transcript, metadata={'url': url})]

                if is_duplicate(texts, vectorstore):
                    print(f"Duplicate skipping")
                else:
                    print(f"NOT A DUPE")
                    try:
                        print("Embedding into Faiss db")
                        vectorstore2 = FAISS.from_documents(texts, embeddings)
                        print("Merging into Faiss db")
                        vectorstore.merge_from(vectorstore2)
                        print("Saving database")
                        vectorstore.save_local(database_path)
                    except Exception as e:
                        print(f"Failed to load or create database due to error: {str(e)}")
            else:
                print(f"No English transcript found for video {url}")
if __name__ == "__main__":
    main()
