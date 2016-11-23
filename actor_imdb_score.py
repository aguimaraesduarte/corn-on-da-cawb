
#--------------------------------------#
# AVERAGE ACTOR SCORE				   #
#--------------------------------------#

actor1 = set(movies['actor_1_name'])
actor2 = set(movies['actor_2_name'])
actor3 = set(movies['actor_3_name'])

actors = np.array(list(actor1.union(actor2).union(actor3)))
actors = actors[actors != 'nan']

def actor_imdb_score(input_list):
	actor_name = []
	mean_score = []

	for i in input_list:
		data1 = movies.loc[movies['actor_1_name'] == i,'imdb_score']
		data2 = movies.loc[movies['actor_2_name'] == i,'imdb_score']
		data3 = movies.loc[movies['actor_3_name'] == i,'imdb_score']

		data_all = data1.append(data2).append(data3)

		actor_name += [i]
		mean_score += [data_all.mean()]

	return actor_name, mean_score

test_actor_name, test_mean_score  = actor_imdb_score(actors)