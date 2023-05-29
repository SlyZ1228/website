# Baseball Pitch Type Prediction

Living in Houston, it would be sacrilegious for me to not be a huge fan of baseball and the Houston Astros. For years, I've cheered, cried, and laughed as the Astros went from a pretty bad team to arguably the best team in the league (no matter what the Yankees say!). We acquired some of the best batters in the league as well as one of the greatest pitching lineups in MLB history. One of these incredible pitchers is Houston's favorite closer, Ryan Pressly. He has come to be known as the ninth-inning pitcher who never lets up a run, which is why I want to attempt to make a predictive algorithm that can determine what pitch Pressly would throw based on the status of the game at the time (ie. the number of balls and strikes, the previous pitch, etc.)


```python
#importing some necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pybaseball as pb
import numpy as np
```


```python
#getting the pitching information for all of Pressly's games for the 2022 season
raw_data = pb.statcast_pitcher('2022-04-07', '2022-10-05', player_id = 519151)
raw_data.info()
```

    Gathering Player Data
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 711 entries, 0 to 710
    Data columns (total 92 columns):
     #   Column                           Non-Null Count  Dtype  
    ---  ------                           --------------  -----  
     0   pitch_type                       711 non-null    object 
     1   game_date                        711 non-null    object 
     2   release_speed                    711 non-null    float64
     3   release_pos_x                    711 non-null    float64
     4   release_pos_z                    711 non-null    float64
     5   player_name                      711 non-null    object 
     6   batter                           711 non-null    int64  
     7   pitcher                          711 non-null    int64  
     8   events                           181 non-null    object 
     9   description                      711 non-null    object 
     10  spin_dir                         0 non-null      float64
     11  spin_rate_deprecated             0 non-null      float64
     12  break_angle_deprecated           0 non-null      float64
     13  break_length_deprecated          0 non-null      float64
     14  zone                             711 non-null    int64  
     15  des                              711 non-null    object 
     16  game_type                        711 non-null    object 
     17  stand                            711 non-null    object 
     18  p_throws                         711 non-null    object 
     19  home_team                        711 non-null    object 
     20  away_team                        711 non-null    object 
     21  type                             711 non-null    object 
     22  hit_location                     165 non-null    float64
     23  bb_type                          104 non-null    object 
     24  balls                            711 non-null    int64  
     25  strikes                          711 non-null    int64  
     26  game_year                        711 non-null    int64  
     27  pfx_x                            711 non-null    float64
     28  pfx_z                            711 non-null    float64
     29  plate_x                          711 non-null    float64
     30  plate_z                          711 non-null    float64
     31  on_3b                            40 non-null     float64
     32  on_2b                            87 non-null     float64
     33  on_1b                            173 non-null    float64
     34  outs_when_up                     711 non-null    int64  
     35  inning                           711 non-null    int64  
     36  inning_topbot                    711 non-null    object 
     37  hc_x                             104 non-null    float64
     38  hc_y                             104 non-null    float64
     39  tfs_deprecated                   0 non-null      float64
     40  tfs_zulu_deprecated              0 non-null      float64
     41  fielder_2                        711 non-null    int64  
     42  umpire                           0 non-null      float64
     43  sv_id                            0 non-null      float64
     44  vx0                              711 non-null    float64
     45  vy0                              711 non-null    float64
     46  vz0                              711 non-null    float64
     47  ax                               711 non-null    float64
     48  ay                               711 non-null    float64
     49  az                               711 non-null    float64
     50  sz_top                           711 non-null    float64
     51  sz_bot                           711 non-null    float64
     52  hit_distance_sc                  192 non-null    float64
     53  launch_speed                     189 non-null    float64
     54  launch_angle                     189 non-null    float64
     55  effective_speed                  711 non-null    float64
     56  release_spin_rate                710 non-null    float64
     57  release_extension                711 non-null    float64
     58  game_pk                          711 non-null    int64  
     59  pitcher.1                        711 non-null    int64  
     60  fielder_2.1                      711 non-null    int64  
     61  fielder_3                        711 non-null    int64  
     62  fielder_4                        711 non-null    int64  
     63  fielder_5                        711 non-null    int64  
     64  fielder_6                        711 non-null    int64  
     65  fielder_7                        711 non-null    int64  
     66  fielder_8                        711 non-null    int64  
     67  fielder_9                        711 non-null    int64  
     68  release_pos_y                    711 non-null    float64
     69  estimated_ba_using_speedangle    104 non-null    float64
     70  estimated_woba_using_speedangle  104 non-null    float64
     71  woba_value                       181 non-null    float64
     72  woba_denom                       181 non-null    float64
     73  babip_value                      181 non-null    float64
     74  iso_value                        181 non-null    float64
     75  launch_speed_angle               104 non-null    float64
     76  at_bat_number                    711 non-null    int64  
     77  pitch_number                     711 non-null    int64  
     78  pitch_name                       711 non-null    object 
     79  home_score                       711 non-null    int64  
     80  away_score                       711 non-null    int64  
     81  bat_score                        711 non-null    int64  
     82  fld_score                        711 non-null    int64  
     83  post_away_score                  711 non-null    int64  
     84  post_home_score                  711 non-null    int64  
     85  post_bat_score                   711 non-null    int64  
     86  post_fld_score                   711 non-null    int64  
     87  if_fielding_alignment            711 non-null    object 
     88  of_fielding_alignment            711 non-null    object 
     89  spin_axis                        710 non-null    float64
     90  delta_home_win_exp               711 non-null    float64
     91  delta_run_exp                    711 non-null    float64
    dtypes: float64(46), int64(29), object(17)
    memory usage: 511.2+ KB
    

After looking at the different types of data available when getting the pitching information, I racked my brain to figure out what factors would be seemingly important to a pitcher in the short time frame they have between pitches. Obviously, the count (how many balls and strikes there are) is something considered by the pitcher. Additionally, the handedness of the batter (right or left-handed) would change the type of pitch a pitcher tends to use. Maybe the pitch number for each at-bat will affect the pitcher's decision? One thing that I wasn't sure about was the last pitch and second to last pitch type; perhaps that might impact the pitcher's choice? Well, to test these theories, let's first do a bit of feature engineering and combine all the features into a singular, organized data frame.


```python
#using the pitch type data to make the last pitch and second to last pitch features
last_pitch = raw_data.get(["pitch_name"]).shift(-1)
last_pitch = last_pitch.mask(last_pitch.eq('None')).dropna()
second_last = raw_data.get(["pitch_name"]).shift(-2)
second_last = second_last.mask(second_last.eq('None')).dropna()
```


```python
#creating the organized dataframe
df = raw_data.get(["game_date", "pitch_name", "pitch_number", "strikes", "balls", "stand"])
df.insert(3, "last_pitch", last_pitch)
df.insert(4, "second_last", second_last)

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game_date</th>
      <th>pitch_name</th>
      <th>pitch_number</th>
      <th>last_pitch</th>
      <th>second_last</th>
      <th>strikes</th>
      <th>balls</th>
      <th>stand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-10-05</td>
      <td>Changeup</td>
      <td>7</td>
      <td>4-Seam Fastball</td>
      <td>Curveball</td>
      <td>2</td>
      <td>3</td>
      <td>R</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-10-05</td>
      <td>4-Seam Fastball</td>
      <td>6</td>
      <td>Curveball</td>
      <td>Slider</td>
      <td>2</td>
      <td>2</td>
      <td>R</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-10-05</td>
      <td>Curveball</td>
      <td>5</td>
      <td>Slider</td>
      <td>Slider</td>
      <td>2</td>
      <td>1</td>
      <td>R</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-10-05</td>
      <td>Slider</td>
      <td>4</td>
      <td>Slider</td>
      <td>Slider</td>
      <td>2</td>
      <td>1</td>
      <td>R</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-10-05</td>
      <td>Slider</td>
      <td>3</td>
      <td>Slider</td>
      <td>Curveball</td>
      <td>1</td>
      <td>1</td>
      <td>R</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>706</th>
      <td>2022-04-07</td>
      <td>Curveball</td>
      <td>1</td>
      <td>4-Seam Fastball</td>
      <td>Slider</td>
      <td>0</td>
      <td>0</td>
      <td>L</td>
    </tr>
    <tr>
      <th>707</th>
      <td>2022-04-07</td>
      <td>4-Seam Fastball</td>
      <td>3</td>
      <td>Slider</td>
      <td>Slider</td>
      <td>0</td>
      <td>2</td>
      <td>R</td>
    </tr>
    <tr>
      <th>708</th>
      <td>2022-04-07</td>
      <td>Slider</td>
      <td>2</td>
      <td>Slider</td>
      <td>4-Seam Fastball</td>
      <td>0</td>
      <td>1</td>
      <td>R</td>
    </tr>
    <tr>
      <th>709</th>
      <td>2022-04-07</td>
      <td>Slider</td>
      <td>1</td>
      <td>4-Seam Fastball</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>R</td>
    </tr>
    <tr>
      <th>710</th>
      <td>2022-04-07</td>
      <td>4-Seam Fastball</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>R</td>
    </tr>
  </tbody>
</table>
<p>711 rows × 8 columns</p>
</div>



Alright, now that I have the organized data frame of my features and labels, I need to encode all of it for it to, you know, actually work with the algorithms! First, however, I removed all of the data that is the first or second pitch of the at-bat. My reasoning behind this is that at-bats act as a sort of refresh for pitchers - a clean slate if you will. Additionally, by shifting the data up by 1 and 2 for the last and second to last pitch data respectively, for the first and second pitch of a game, the data would say that the last pitch and second last pitch would be from the last game, which just doesn't make sense! Finally, I'm going to combine two of the label types, Curveballs and Sliders, for two reasons. One, when grouping pitch types, people usually combine them as one category - Breaking Balls. The second reason is more important in terms of practicality; the pitches that are most important for batters to recognize (for them to hit it well) are fastballs and changeups, which means that differentiating between fastballs, changeups, and breaking balls is the most practical setup for this algorithm.


```python
#combining the curveballs and sliders

df = df.mask(df['pitch_number'].eq(1)).dropna()
df = df.mask(df['pitch_number'].eq(2)).dropna()

df['pitch_name'] = df['pitch_name'].replace('4-Seam Fastball', 'fast')
df['pitch_name'] = df['pitch_name'].replace(['Curveball', 'Slider'], 'breaking')


df['last_pitch'] = df['last_pitch'].replace('4-Seam Fastball', 'fast')
df['last_pitch'] = df['last_pitch'].replace(['Curveball', 'Slider'], 'breaking')

df['second_last'] = df['second_last'].replace('4-Seam Fastball', 'fast')
df['second_last'] = df['second_last'].replace(['Curveball', 'Slider'], 'breaking')

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game_date</th>
      <th>pitch_name</th>
      <th>pitch_number</th>
      <th>last_pitch</th>
      <th>second_last</th>
      <th>strikes</th>
      <th>balls</th>
      <th>stand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-10-05</td>
      <td>Changeup</td>
      <td>7.0</td>
      <td>fast</td>
      <td>breaking</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>R</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-10-05</td>
      <td>fast</td>
      <td>6.0</td>
      <td>breaking</td>
      <td>breaking</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>R</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-10-05</td>
      <td>breaking</td>
      <td>5.0</td>
      <td>breaking</td>
      <td>breaking</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>R</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-10-05</td>
      <td>breaking</td>
      <td>4.0</td>
      <td>breaking</td>
      <td>breaking</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>R</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-10-05</td>
      <td>breaking</td>
      <td>3.0</td>
      <td>breaking</td>
      <td>breaking</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>R</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>699</th>
      <td>2022-04-10</td>
      <td>breaking</td>
      <td>4.0</td>
      <td>fast</td>
      <td>fast</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>R</td>
    </tr>
    <tr>
      <th>700</th>
      <td>2022-04-10</td>
      <td>fast</td>
      <td>3.0</td>
      <td>fast</td>
      <td>fast</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>R</td>
    </tr>
    <tr>
      <th>703</th>
      <td>2022-04-07</td>
      <td>breaking</td>
      <td>4.0</td>
      <td>fast</td>
      <td>fast</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>L</td>
    </tr>
    <tr>
      <th>704</th>
      <td>2022-04-07</td>
      <td>fast</td>
      <td>3.0</td>
      <td>fast</td>
      <td>breaking</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>L</td>
    </tr>
    <tr>
      <th>707</th>
      <td>2022-04-07</td>
      <td>fast</td>
      <td>3.0</td>
      <td>breaking</td>
      <td>breaking</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>R</td>
    </tr>
  </tbody>
</table>
<p>367 rows × 8 columns</p>
</div>




```python
#encoding yayyy woo hoo

df["pitch_name"] = df["pitch_name"].astype('category')
df["pitch_name"] = df["pitch_name"].cat.codes

df["last_pitch"] = df["last_pitch"].astype('category')
df["last_pitch"] = df["last_pitch"].cat.codes

df["second_last"] = df["second_last"].astype('category')
df["second_last"] = df["second_last"].cat.codes

df['strikes'] = df['strikes'].astype(int)
df['balls'] = df['balls'].astype(int)
df['pitch_number'] = df['pitch_number'].astype(int)

df['stand'] = df['stand'].astype('category')
df['stand'] = df['stand'].cat.codes

df = df.drop(columns = 'game_date')

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pitch_name</th>
      <th>pitch_number</th>
      <th>last_pitch</th>
      <th>second_last</th>
      <th>strikes</th>
      <th>balls</th>
      <th>stand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>699</th>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>700</th>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>703</th>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>704</th>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>707</th>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>367 rows × 7 columns</p>
</div>



Alright, our data is set! Now I'll apply the data to the models. There are many classification algorithms that we can use, and there are two main types that I want to use in this project - an SVM (support vector machine) as well as a Random Forest Classifier. So let's create both of those models using scikit-learn and print the respective accuracy scores of the two models for their predictions of the test data.


```python
#split data into train and test sets

X = np.array(df.drop(columns = 'pitch_name'))
y = np.array(df['pitch_name'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=61)
```


```python
#first model - random forest model

clf = RandomForestClassifier(n_estimators = 200)
clf.fit(X_train, y_train)

clf.score(X_test, y_test)
```




    0.6195652173913043




```python
#next model - svm

rbf = svm.SVC(kernel='rbf', gamma=10, C=1).fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=2, C=0.14).fit(X_train, y_train)

poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)
```


```python
#check accuracy of the polynomial and rbf svm

poly_accuracy = accuracy_score(y_test, poly_pred)
poly_f1 = f1_score(y_test, poly_pred, average='weighted')
print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

rbf_accuracy = accuracy_score(y_test, rbf_pred)
rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))
```

    Accuracy (Polynomial Kernel):  76.09
    F1 (Polynomial Kernel):  71.82
    Accuracy (RBF Kernel):  71.74
    F1 (RBF Kernel):  71.62
    

Looking at the accuracy scores of the SVMs and the random forest classifier, it would appear that, after hyperparameter tuning, the polynomial kernel SVM had the best accuracy of around 76.09%, but the RBF kernel and Random Forest model both had accuracies that were lagging behind only by a little bit. Creating a binary set of labels by combining all other pitch types besides fastballs into one category was something that crossed my mind while trying to increase the accuracy of the model. However, when testing that possibility, I found that, even though the base accuracy (without hyperparameter tuning) increased, after hyperparameter tuning, the accuracy (~71)% was around the same as the current dataset. Therefore, I stuck with the three labels shown (Fastballs, Changeups, and Breaking Balls) because, as stated before, the pitches that are most important for batters to recognize (for them to hit it well) are fastballs and changeups, which makes this the ideal data setup.

Lastly, a note about the practicality of this model: I'm sure many people who aren't avid fans of baseball or who haven't played baseball before would question whether or not this model has any use in the real world. I wholeheartedly believe that this model and others similar to it could be extremely helpful to teams and individual players who want to increase their hitting average. Knowing the type of pitch coming while you're up to bat is extremely helpful to players as seen by the many sign-stealing scandals throughout MLB history, which essentially did the same as this model - the sign-stealing plots gave batters information about the pitch coming that they would have never gotten until the pitch was actually over with. If players can learn or at least approximate the patterns found by models such as this one, they can semi-accurately predict what pitch is coming next and hit it out of the park!
