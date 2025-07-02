# Predicting Playlists

## Summary 
A key challenge in music recommendation is suggesting songs that match the overall ‘vibe’ of a user’s playlist, which often includes hidden groupings to create a broader tone. Using collaborative and content-based filtering approaches, we built and compared recommendation models that use K-means clustering to predict missing tracks and suggest new songs that fit seamlessly into existing playlists.

## Models 
+ Model #0 (Baseline): Recommends songs that simply co-occurred with any song in the playlist, selecting 5 random songs from this pool.
+ Model #1 (Co-occurrence Similarity): Obtains playlist-level co-occurrence patterns by analyzing how songs tend to appear together across many playlists. Uses K-means clustering on 75% of the playlist to group similar songs, then recommends 5 songs with the highest cosine similarity to the 75%-playlist’s co-occurrence clusters.
+ Model #2 (Emotion Similarity): Extracts the emotional profile of playlists using an pre-existing natural language processing model (Distilbert) trained on the GoEmotions dataset. Uses K-means clustering of 75% of each playlist's songs based on emotional vectors, and then recommends 5 songs with the highest similarity to the 75%-playlist’s emotional clusters. 

## Guide to the repository
*Files*  
/playlist_subset/playlist_subset_*.csv.gz: Contains the playlist data. 

/playlist_subset/freq/song_frequency_csv.gz: Includes the frequency counts for songs across playlists. 

/playlist_subset/lyrics/all_lyrics_final.gz: Consists of the lyrics extracted via the Genius API. 

/playlist_subset/lyrics/emo/lyrics_27_goemotions.csv.zip: Includes lyrics-derived extracted emotion vectors for songs. 

*Scripts*  
extract_lyrics.ipynb: Extracts lyrics for songs that appeared with the most frequency (>= 20 playlists) across the entire dataset. This is primarily completed via Genius.com using the python package LyricsGenius. 

extract_emo_sentiment.ipynb: Uses an existing natural language processing model (FILL) trained on the GoEmotions Google dataset. This model extracts 28 emotions for songs that contain lyrics. 

split_dataset.ipynb: Splits the playlist dataset (20,000 playlists total) into train-test according to a 80-20% split. 

file_manipulation.ipynb: Computes descriptive statistics on the playlist dataset (e.g., average # of songs in a playlist, unique # of artists per playlist)

baseline_model.ipynb: Contains the code for Model #0 (Baseline), which generates recommendations based on pairwise song co-occurrences. 

models_1and2.ipynb: Contains the code for Model #1 (Co-Occurrence cosine similarity and Model #2 (Emotion cosine similarity)..



