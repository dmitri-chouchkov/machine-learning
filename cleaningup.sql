-- DELETE FROM movies
-- LOAD DATA INFILE 'C:\\Users\\Dmitri\\Desktop\\Recommender Systems\\Low Rank Matrix Completion\\movies.csv' INTO TABLE movies
--  FIELDS TERMINATED BY ',' ENCLOSED BY '"' ESCAPED BY '\\'
-- LINES TERMINATED BY '\n' IGNORE 1 LINES;

-- DELETE FROM ratings
-- LOAD DATA INFILE 'C:\\Users\\Dmitri\\Desktop\\Recommender Systems\\Low Rank Matrix Completion\\ratings.csv' INTO TABLE ratings
-- FIELDS TERMINATED BY ',' ENCLOSED BY '"' ESCAPED BY '\\'
-- LINES TERMINATED BY '\n' IGNORE 1 LINES;

-- alter table movies add column row_num int
-- alter table ratings add column movie_row_num int
-- alter table ratings add column zscore double

-- CREATE TEMPORARY TABLE IF NOT EXISTS users as
-- (select userId as userId, avg(rating) as mean, std(rating) as stdev from ratings group by userId order by userId asc)
-- update users set stdev = 1 where stdev = 0

-- update ratings join users on users.userId = ratings.userId
-- set ratings.zscore = (ratings.rating - users.mean)/users.stdev

-- update movies join
-- (SELECT  movies.movieId,  @row_num:= @row_num + 1 AS Therow FROM movies, (SELECT @row_num:= 0 AS num) AS c ORDER BY movies.movieId ASC) mdata on movies.movieId = mdata.movieId
-- set movies.row_num = mdata.Therow

-- alter table ratings add column movie_row_num int

-- update ratings join
-- movies on movies.movieId = ratings.movieId
--  set ratings.movie_row_num = movies.row_num;

-- CREATE TEMPORARY TABLE IF NOT EXISTS users as
-- (select userId as userId, avg(rating) as mean, std(rating) as stdev from ratings group by userId order by userId asc)
-- update users set stdev = 1 where stdev = 0

-- update ratings join users on users.userId = ratings.userId
-- set ratings.zscore = (ratings.rating - users.mean)/users.stdev

-- generate new csv files that don't suck

 -- select * from ratings ORDER BY userId, movieId
 --   INTO OUTFILE 'C:\\Users\\Dmitri\\Desktop\\Recommender Systems\\Low Rank Matrix Completion\\ratings_clean.csv'
 --   FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' ESCAPED BY '\\'
 --   LINES TERMINATED BY '\n';

-- select * from movies ORDER BY movieId
--    INTO OUTFILE 'C:\\Users\\Dmitri\\Desktop\\Recommender Systems\\Low Rank Matrix Completion\\movies_clean.csv'
--    FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"' ESCAPED BY '\\'
--    LINES TERMINATED BY '\n';


-- select * from movies

-- SELECT  movies.*,  @row_num:= @row_num + 1 AS Therow FROM movies, (SELECT @row_num:= 0 AS num) AS c ORDER BY movies.movieId ASC;

-- SELECT  users.userId,  @row_num:= @row_num + 1 AS Therow FROM (select distinct userId from ratings) as users, (SELECT @row_num:= 0 AS num) AS c ORDER BY users.userId ASC;  
