"""
extract_reddit.py - Historical Reddit extractor (Arctic Shift + PRAW)

Fetches "Weekend Discussion Thread" and "Daily Discussion Thread" posts from a
subreddit via the Arctic Shift API, keeps the most-commented thread per day,
and downloads their comments with PRAW.

Reddit API credentials are read from the environment:
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

Output: <subreddit>_Discussion_Threads_<start>_to_<end>_{posts,comments}.csv

Reproduction (values used in the study): subreddit = wallstreetbets,
comment_limit = 50; run once per year for 2022-01-01..2022-12-31 and
2023-01-01..2023-12-31. The prompts below default to these values
(press Enter to accept); only the date range must be entered each run.
"""

import os
import sys
import time
from datetime import datetime

import praw
import requests
import pandas as pd

# Optionally load credentials from a local .env file (never committed).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def format_time(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs:02d}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins:02d}m"


def fetch_posts_arcticshift(subreddit_name, search_query, fecha_inicio, fecha_fin):
    """Fetch posts matching a title query from the Arctic Shift API."""
    base_url = "https://arctic-shift.photon-reddit.com/api/posts/search"
    all_posts = []
    current_after = fecha_inicio

    consecutive_errors = 0
    max_errors = 3
    empty_responses = 0
    wait_time = 2

    print(f"\n  Searching: '{search_query}'")

    while consecutive_errors < max_errors:
        try:
            params = {
                'subreddit': subreddit_name,
                'title': search_query,
                'after': current_after,
                'before': fecha_fin,
                'limit': 100,
                'sort': 'asc',
                'sort_type': 'created_utc'
            }

            response = requests.get(base_url, params=params, timeout=30)

            if response.status_code == 422:
                wait_time = min(wait_time * 2, 30)
                print(f"    Server timeout, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            if response.status_code == 429:
                print(f"    Rate limit reached, waiting 60s...")
                time.sleep(60)
                wait_time = 5
                continue

            if response.status_code == 500:
                consecutive_errors += 1
                print(f"    Error 500, waiting... ({consecutive_errors}/{max_errors})")
                time.sleep(5)
                continue

            if response.status_code != 200:
                print(f"    Error {response.status_code}: {response.text}")
                consecutive_errors += 1
                time.sleep(5)
                continue

            data = response.json()
            posts = data.get('data', [])

            if not posts:
                empty_responses += 1
                if empty_responses >= 2:
                    break
                time.sleep(2)
                continue

            empty_responses = 0
            consecutive_errors = 0

            if wait_time > 2:
                wait_time = max(wait_time * 0.8, 2)

            for post in posts:
                post_info = {
                    'post_id': post['id'],
                    'date': datetime.fromtimestamp(post['created_utc']).strftime('%Y-%m-%d'),
                    'time': datetime.fromtimestamp(post['created_utc']).strftime('%H:%M:%S'),
                    'datetime_full': datetime.fromtimestamp(post['created_utc']).strftime('%Y-%m-%d %H:%M:%S'),
                    'title': post['title'],
                    'author': post.get('author', '[deleted]'),
                    'score': post.get('score', 0),
                    'upvote_ratio': post.get('upvote_ratio', 0),
                    'num_comments': post.get('num_comments', 0),
                    'num_comments_downloaded': 0,
                    'created_utc': post['created_utc'],
                    'url': f"https://reddit.com{post['permalink']}",
                    'search_term': search_query
                }
                all_posts.append(post_info)

            if len(all_posts) % 100 == 0:
                print(f"    Posts found: {len(all_posts)}")

            last_post_time = datetime.fromtimestamp(posts[-1]['created_utc'])
            current_after = (last_post_time + pd.Timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S')

            time.sleep(wait_time)

        except requests.exceptions.Timeout:
            consecutive_errors += 1
            print(f"    Network timeout... ({consecutive_errors}/{max_errors})")
            time.sleep(5)
        except Exception as e:
            print(f"    Error: {e}")
            consecutive_errors += 1
            time.sleep(5)

    print(f"    {len(all_posts)} posts extracted")
    return all_posts


def scrape_reddit_arcticshift():
    """Run the full extraction: posts -> daily filtering -> comments."""
    print("=" * 70)
    print("HISTORICAL REDDIT EXTRACTOR (Arctic Shift + PRAW)")
    print("=" * 70)

    subreddit_name = input("\n1. Subreddit [wallstreetbets]: ").strip() or "wallstreetbets"

    print("\n2. Date range:")
    fecha_inicio = input("   Start date (YYYY-MM-DD): ").strip()
    fecha_fin = input("   End date (YYYY-MM-DD): ").strip()

    datetime.strptime(fecha_inicio, '%Y-%m-%d')
    datetime.strptime(fecha_fin, '%Y-%m-%d')

    print("\n3. Comment expansion:")
    print("   0  = do not extract comments")
    print("   -1 = ALL comments (very slow)")
    print("   N  = limit to N expansions (20-50 recommended)")
    comment_limit = input("   Choose (0, -1, or number) [50]: ").strip() or "50"

    if comment_limit == "-1":
        comment_limit = None
    elif comment_limit == "0":
        comment_limit = 0
    else:
        comment_limit = int(comment_limit)

    print("\n" + "=" * 70)
    print("CONFIGURATION:")
    print(f"  Subreddit: r/{subreddit_name}")
    print(f"  Queries: Weekend Discussion Thread, Daily Discussion Thread")
    print(f"  Period: {fecha_inicio} to {fecha_fin}")
    print(f"  Filtering: per day, keep the thread with MOST COMMENTS")
    print(f"  Comments: {'None' if comment_limit == 0 else 'ALL' if comment_limit is None else f'limit {comment_limit}'}")
    print("=" * 70)

    if input("\nProceed? (y/n): ").lower() != 'y':
        print("Cancelled")
        return None, None

    print(f"\n{'=' * 70}")
    print("PHASE 1: Fetching posts with the Arctic Shift API")
    print("=" * 70)

    all_posts = []
    weekend_posts = fetch_posts_arcticshift(subreddit_name, "Weekend Discussion Thread", fecha_inicio, fecha_fin)
    all_posts.extend(weekend_posts)
    daily_posts = fetch_posts_arcticshift(subreddit_name, "Daily Discussion Thread", fecha_inicio, fecha_fin)
    all_posts.extend(daily_posts)

    if not all_posts:
        print("\nNo posts found")
        return None, None

    print(f"\nTotal posts extracted: {len(all_posts)}")
    print(f"  - Weekend Discussion: {len(weekend_posts)}")
    print(f"  - Daily Discussion: {len(daily_posts)}")

    print(f"\n{'=' * 70}")
    print("PHASE 2: Removing duplicates")
    print("=" * 70)

    posts_df = pd.DataFrame(all_posts)
    print(f"Posts before filtering: {len(posts_df)}")

    # Keep the most-commented thread per day
    posts_filtered = posts_df.sort_values('num_comments', ascending=False).groupby('date').first().reset_index()

    print(f"Posts after filtering: {len(posts_filtered)}")
    print(f"Posts removed: {len(posts_df) - len(posts_filtered)}")

    if len(posts_filtered) > 0:
        print("\nSample of selected posts:")
        for idx, row in posts_filtered.head(5).iterrows():
            print(f"  {row['date']}: {row['title'][:50]}... ({row['num_comments']} comments)")

    base_filename = f"{subreddit_name}_Discussion_Threads_{fecha_inicio}_to_{fecha_fin}"
    posts_filtered.to_csv(f'{base_filename}_posts_FILTRADO.csv', index=False, encoding='utf-8')

    if comment_limit == 0:
        print("\nSkipping comment extraction")
        comments_df = pd.DataFrame()
    else:
        print(f"\n{'=' * 70}")
        print("PHASE 3: Extracting comments with PRAW (filtered posts only)")
        print("=" * 70)

        client_id = os.environ.get("REDDIT_CLIENT_ID")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
        if not client_id or not client_secret:
            print("ERROR: set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET "
                  "(see .env.example). Comment extraction needs PRAW credentials.")
            return None, None

        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=os.environ.get("REDDIT_USER_AGENT", "Reddit_ArcticShift_Scraper"),
        )

        comments_data = []
        start_time = time.time()
        thread_times = []

        for idx, row in posts_filtered.iterrows():
            thread_idx = idx + 1
            thread_start = time.time()

            if len(thread_times) > 0:
                avg_time = sum(thread_times) / len(thread_times)
                remaining = len(posts_filtered) - thread_idx
                eta_str = format_time(avg_time * remaining)
            else:
                eta_str = "estimating..."

            percent = thread_idx / len(posts_filtered)
            filled = int(40 * percent)
            bar = '#' * filled + '-' * (40 - filled)
            sys.stdout.write(f'\r  [{bar}] {percent*100:.1f}% | {thread_idx}/{len(posts_filtered)} threads | ETA: {eta_str}')
            sys.stdout.flush()

            print(f"\n[{thread_idx}] {row['date']} - '{row['title'][:60]}...'")
            print(f"    Score: {row['score']} | Comments: {row['num_comments']}")

            try:
                submission = reddit.submission(id=row['post_id'])
                print(f"    Expanding comments (limit={comment_limit if comment_limit else 'ALL'})...", end=" ")
                submission.comments.replace_more(limit=comment_limit)

                comment_count = 0
                for comment in submission.comments.list():
                    if hasattr(comment, 'body') and comment.body != '[deleted]':
                        comment_datetime = datetime.fromtimestamp(comment.created_utc)
                        comments_data.append({
                            'post_id': row['post_id'],
                            'post_date': row['date'],
                            'comment_id': comment.id,
                            'parent_id': comment.parent_id,
                            'author': str(comment.author),
                            'body': comment.body,
                            'score': comment.score,
                            'created_utc': comment.created_utc,
                            'comment_date': comment_datetime.strftime('%Y-%m-%d'),
                            'comment_time': comment_datetime.strftime('%H:%M:%S'),
                            'comment_datetime_full': comment_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                            'is_submitter': comment.is_submitter
                        })
                        comment_count += 1

                posts_filtered.at[idx, 'num_comments_downloaded'] = comment_count
                print(f"{comment_count}/{row['num_comments']} extracted")

            except Exception as e:
                print(f"Error: {str(e)}")

            thread_times.append(time.time() - thread_start)

        bar = '#' * 40
        total_time = time.time() - start_time
        sys.stdout.write(f'\r  [{bar}] 100.0% | {len(posts_filtered)}/{len(posts_filtered)} threads | Done in {format_time(total_time)}')
        sys.stdout.flush()
        print("\n")

        comments_df = pd.DataFrame(comments_data)

    print("=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    print(f"  Total posts extracted: {len(posts_df)}")
    print(f"  Posts after filtering: {len(posts_filtered)}")
    print(f"  Comments extracted: {len(comments_df)}")
    if not posts_filtered.empty:
        print(f"  Range: {posts_filtered['date'].min()} to {posts_filtered['date'].max()}")

    csv_posts = f'{base_filename}_posts.csv'
    csv_comments = f'{base_filename}_comments.csv'
    posts_filtered.to_csv(csv_posts, index=False, encoding='utf-8')
    if not comments_df.empty:
        comments_df.to_csv(csv_comments, index=False, encoding='utf-8')

    print(f"\nFiles saved:")
    print(f"  - {csv_posts}")
    if not comments_df.empty:
        print(f"  - {csv_comments}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return posts_filtered, comments_df


if __name__ == "__main__":
    posts, comments = scrape_reddit_arcticshift()
