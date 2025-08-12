
from tqdm import tqdm
import pandas as pd
import requests
import re
import json
import os
from urllib.parse import urlparse, parse_qs
from io import StringIO
from bs4 import BeautifulSoup
from typing import Optional, List, Dict, Any, Tuple

class GoogleSheetsReader:
    """A class to read Google Sheets from URLs using different methods."""
    
    def __init__(self, credentials_file: Optional[str] = None):

        self.credentials_file = credentials_file
        
    def extract_sheet_id_from_url(self, url: str) -> str:

        patterns = [
            r'/spreadsheets/d/([a-zA-Z0-9-_]+)',  # Standard format
            r'/d/([a-zA-Z0-9-_]+)',  # Shortened format
            r'id=([a-zA-Z0-9-_]+)'   # Query parameter format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract sheet ID from URL: {url}")
    
    def extract_gid_from_url(self, url: str) -> Optional[str]:
        gid_patterns = [
            r'[?&]gid=(\d+)',  # Query parameter format - fixed: removed extra backslash
            r'#gid=(\d+)'      # Hash format - fixed: removed extra backslash
        ]
        
        for pattern in gid_patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def read_sheet_as_html(self, url: str, sheet_name: Optional[str] = None, gid: Optional[str] = None) -> pd.DataFrame:

        try:
            sheet_id = self.extract_sheet_id_from_url(url)
            
            # Extract gid from URL if not provided
            if gid is None:
                gid = self.extract_gid_from_url(url)
            
            if gid:
                html_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:html&gid={gid}"
            elif sheet_name:
                import urllib.parse
                encoded_sheet_name = urllib.parse.quote(sheet_name)
                html_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:html&sheet={encoded_sheet_name}"
            else:
                html_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:html"
            
            # Read the HTML data
            response = requests.get(html_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            table = soup.find('table')
            if not table:
                raise Exception("No table found in HTML response")
            
            headers = []
            header_row = table.find('tr')
            if header_row:
                for th in header_row.find_all('th'):
                    headers.append(th.get_text(strip=True))
            
            # If no headers found, try to get them from the first data row
            if not headers:
                first_data_row = table.find_all('tr')[1] if len(table.find_all('tr')) > 1 else None
                if first_data_row:
                    for td in first_data_row.find_all('td'):
                        headers.append(td.get_text(strip=True))
            
            data = []
            data_rows = table.find_all('tr')[1:] if headers else table.find_all('tr')
            
            for row in data_rows:
                row_data = []
                
                for td in row.find_all('td'):
                    text = td.get_text(strip=True)
                    row_data.append(text)
                
                if row_data:
                    data.append(row_data)
            
            max_cols = max(len(row) for row in data) if data else 0
            
            for row in data:
                while len(row) < max_cols:
                    row.append('')
            
            # Create headers if none exist or if we need more
            if not headers:
                headers = [f'Column_{i+1}' for i in range(max_cols)]
            elif len(headers) < max_cols:
                for i in range(len(headers), max_cols):
                    headers.append(f'Column_{i+1}')
            elif len(headers) > max_cols:
                headers = headers[:max_cols]
            
            df = pd.DataFrame(data, columns=headers)
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to read sheet as HTML: {str(e)}")
    
    def read_sheet_as_csv(self, url: str, sheet_name: Optional[str] = None, gid: Optional[str] = None) -> pd.DataFrame:

        try:
            sheet_id = self.extract_sheet_id_from_url(url)
            
            # Extract gid from URL if not provided
            if gid is None:
                gid = self.extract_gid_from_url(url)
            
            if gid:
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid}"
            elif sheet_name:
                import urllib.parse
                encoded_sheet_name = urllib.parse.quote(sheet_name)
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={encoded_sheet_name}"
            else:
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
            
            response = requests.get(csv_url)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text))
            return df
            
        except Exception as e:
            raise Exception(f"Failed to read sheet as CSV: {str(e)}")
    

    def read_sheet(self, url: str, method: str = 'csv', sheet_name: Optional[str] = None, 
                gid: Optional[str] = None) -> pd.DataFrame:
        if method == 'csv':
            df = self.read_sheet_as_csv(url, sheet_name, gid)
        elif method == 'html':
            df = self.read_sheet_as_html(url, sheet_name, gid)

        else:
            raise ValueError("Method must be 'csv', 'html', 'gspread', or 'api'")
        
        return df


def read_google_sheet(url: str, method: str = 'csv', sheet_name: Optional[str] = None, 
                     credentials_file: Optional[str] = None, gid: Optional[str] = None) -> pd.DataFrame:
    reader = GoogleSheetsReader(credentials_file)
    return reader.read_sheet(url, method, sheet_name, gid)



def extract_file_id_from_url(gdrive_url: str) -> Optional[str]:

    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',  # Standard file URL
        r'/open\?id=([a-zA-Z0-9_-]+)',  # Open URL
        r'/uc\?id=([a-zA-Z0-9_-]+)',   # Direct download URL
    ]
    
    for pattern in patterns:
        match = re.search(pattern, gdrive_url)
        if match:
            return match.group(1)
    
    return None


def read_gdrive_file_as_text(gdrive_url: str, encoding: str = 'utf-8') -> str:

    file_id = extract_file_id_from_url(gdrive_url)
    if not file_id:
        raise ValueError(f"Could not extract file ID from URL: {gdrive_url}")
    
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()
        content = response.content.decode(encoding)
        return content
        
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to download file: {e}")
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(f"Failed to decode file with encoding '{encoding}': {e}")


def parse_srt_content(srt_content: str) -> List[Dict[str, Any]]:

    # Split content into subtitle blocks
    subtitle_blocks = re.split(r'\n\s*\n', srt_content.strip())
    
    subtitles = []
    
    for block in subtitle_blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
            
        try:
            index = int(lines[0])
            
            timestamp_line = lines[1]
            time_match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', timestamp_line)
            
            if not time_match:
                continue
                
            start_h, start_m, start_s, start_ms = map(int, time_match.groups()[:4])
            end_h, end_m, end_s, end_ms = map(int, time_match.groups()[4:])
            
            start_seconds = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000.0
            end_seconds = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000.0
            
            text = '\n'.join(lines[2:]).strip()
            
            subtitles.append({
                'index': index,
                'start_seconds': start_seconds,
                'end_seconds': end_seconds,
                'text': text
            })
            
        except (ValueError, IndexError):
            continue
    
    return subtitles


def extract_metadata_from_row(row: pd.Series) -> Dict[str, Any]:

    metadata = {}
    
    if 'Name' in row.index:
        metadata['title'] = str(row['Name'])
    
    if 'URL' in row.index:
        metadata['video_url'] = str(row['URL'])
    
    if 'Summary' in row.index:
        metadata['summary'] = str(row['Summary'])

    
    return metadata


def parse_gdrive_srt_with_metadata(
    gdrive_url: str, 
    row: pd.Series,
    encoding: str = 'utf-8'
) -> Tuple[List[str], List[Dict[str, Any]]]:
    srt_content = read_gdrive_file_as_text(gdrive_url, encoding)

    subtitles = parse_srt_content(srt_content)

    main_metadata = extract_metadata_from_row(row)
    
    # print(main_metadata)
    text_list = [subtitle['text'] for subtitle in subtitles]
    

    metadata_list = []
    for subtitle in subtitles:
        subtitle_metadata = {
            'start_seconds': subtitle['start_seconds'],
            'end_seconds': subtitle['end_seconds'],
            'index': subtitle['index'],
            # 'type': 'video',
            **main_metadata  # Include main metadata (name, video_url, summary)
        }
        metadata_list.append(subtitle_metadata)
    
    return text_list, metadata_list


def proc_BDC_vids_Google_Sheet(url: str, method: str = 'html', transcript_col: str = 'Transcript (with timestamps)'):
    # Read the Google Sheet
    # sheet_url = "https://docs.google.com/spreadsheets/d/1vUVMffOGz3Eggu4RjZSjSRToQ3Ydc2uHxCjSDdOpvug/edit?gid=397146063#gid=397146063"
    
    all_metadata = []
    all_text = []
    
    
    try:
        # Read the sheet using the html method
        df = read_google_sheet(url, method=method)
        print(f"Successfully read sheet with {len(df)} rows and {len(df.columns)} columns")
        
        
        if transcript_col in df.columns:
            # drop rows where the URL column is string with len=0 or nan
            df = df[df[transcript_col].str.len() > 0]
            
            # Filter out header rows (rows where the first column contains "Name" or similar)
            df = df[~df.iloc[:, 0].str.contains('Name|name', na=False)]
            df = df.reset_index(drop=True)
            
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                gdrive_url = row[transcript_col]
                # print(row)
                if pd.notna(gdrive_url) and 'drive.google.com' in str(gdrive_url):
                    try:
                        text_list, metadata_list = parse_gdrive_srt_with_metadata(gdrive_url, row)
                        all_text.append(text_list)
                        all_metadata.append(metadata_list)

                    except Exception as e:
                        print(f"Error processing row {idx}: {e}")
                else:
                    print(f"\nRow {idx}: No valid Google Drive URL found: {row[transcript_col]}")
        else:
            print(f"Column '{transcript_col}' not found. Available columns: {df.columns.tolist()}")
            
    except Exception as e:
        print(f"Error reading sheet: {e}")

    return all_text, all_metadata


