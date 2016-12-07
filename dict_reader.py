"""
This script reads in a txt file where each line in the file is a dictionary in Python dictionary format.
The contents of this file are are first stored in a list-of-lists which is used to create a Panda's data
frame. This data frame is the merged with the IMDB 5000 movie data. This gives more information for use
in modeling on each movie.

"""


from datetime import datetime
import pandas as pd


def create_df_from_dict_list(d_list):
    index = range(0, len(d_list))
    df = pd.DataFrame(d_list, index=index)
    return df


def extract_month(date_string):
    try:
        date_obj = datetime.strptime(date_string, '%d %b %Y')
        return date_obj.month
    except (TypeError, ValueError):
        return date_string


def read_format_imdb_5000():
    df = pd.read_table("movie_metadata.csv", sep=",")
    df['movie_title'] = df['movie_title'].apply(lambda x: x.replace('\xc2\xa0', ''))
    return df


def read_file_to_dict_list(file_name):
    d_list = []
    with open(file_name, 'r') as f:
        for line in f:
            raw_dict = eval(line)
            try:
                raw_dict['title']
                d_list.append(raw_dict)
            except KeyError:
                # if the title field is blank, we do not append the data
                pass
    return d_list


if __name__ == '__main__':

    dict_list = read_file_to_dict_list('imdb_movie_dictionaries.txt')
    imdb = create_df_from_dict_list(dict_list)

    # Uncomment this line to write raw data to a file.
    # imdb.to_csv('imbd_raw_data.csv', encoding='UTF-8', index=False)

    # this line reads the original IMDB 5000 movie data
    movies = read_format_imdb_5000()

    # keep only the new fields we might use and title for the join/merge.
    imdb = imdb[['title', 'type', 'awards', 'plot', 'metascore', 'writer', 'released']]
    imdb.rename(columns={'title': 'movie_title'}, inplace=True)

    combo_df = pd.merge(movies, imdb, how='left', on='movie_title')

    combo_df['release_month'] = combo_df['released'].apply(extract_month)

    # Uncomment this line to write the final csv for use in modeling
    # combo_df.to_csv('movie_data.csv', encoding='UTF-8', index=False)