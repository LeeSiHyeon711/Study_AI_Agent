# YoutubeLoader 대신 직접 youtube_transcript_api 사용하는 방법
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_search import YoutubeSearch
import re

def extract_video_id(url):
    """URL에서 video_id를 추출하는 함수"""
    pattern = r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_youtube_transcript(video_url, languages=['ko', 'en']):
    """유튜브 동영상의 자막을 가져오는 함수"""
    video_id = extract_video_id(video_url)
    print(f"Video ID: {video_id}")
    
    try:
        # YouTubeTranscriptApi 인스턴스 생성
        yt_api = YouTubeTranscriptApi()
        
        # 지정된 언어로 자막 가져오기
        transcript = yt_api.fetch(video_id, languages=languages)
        print("자막을 성공적으로 가져왔습니다!")
        print(f"총 {len(transcript)}개의 자막 세그먼트가 있습니다.")
        return transcript
        
    except Exception as e:
        print(f"오류 발생: {e}")
        
        # 다른 언어로 시도
        try:
            yt_api = YouTubeTranscriptApi()
            transcript = yt_api.fetch(video_id, languages=['en'])
            print("영어 자막으로 성공!")
            return transcript
        except Exception as e2:
            print(f"영어 자막도 실패: {e2}")
            return None

# 사용 예시
if __name__ == "__main__":
    # 키워드 검색
    videos = YoutubeSearch("미국 대선", max_results=5).to_dict()
    
    # 온전한 경로 만들기
    video_url = 'https://youtube.com' + videos[3]['url_suffix']
    print(f"선택된 비디오 URL: {video_url}")
    
    # 자막 가져오기
    transcript = get_youtube_transcript(video_url)
    
    if transcript:
        # 첫 번째 자막 세그먼트 확인
        print("\n첫 번째 자막 세그먼트:")
        print(f"텍스트: {transcript[0].text}")
        print(f"시작 시간: {transcript[0].start}")
        print(f"지속 시간: {transcript[0].duration}")
        
        # 전체 자막 텍스트 합치기
        full_text = " ".join([segment.text for segment in transcript])
        print(f"\n전체 자막 길이: {len(full_text)} 문자")
        print(f"자막 미리보기: {full_text[:200]}...")
