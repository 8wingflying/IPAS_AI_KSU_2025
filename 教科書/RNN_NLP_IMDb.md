# 🎬IMDb 網路電影資料集
- TExt Classification|文本辨識(分類)
- 情感分析(Sentiment Analysis)
  - 情感分析是一個計算上的過程，用於識別和分類在文本數據中表達的意見、情感和態度。
  - 它利用自然語言處理 (NLP) 技術和機器學習算法來分析文本內容，將其分類為積極、消極或中性。 
# 推薦系統(Recommendation System)
- [IMDB Movies Dataset|Top 1000 Movies by IMDB Rating](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)
# 推薦系統
- [E-Commerce Data@Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
  - https://www.kaggle.com/code/abdullahasiff/data-analysis-and-exploration 
- 🎥[Netflix Prize data - Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
  - Netflix Prize百萬美金競賽
  - 2019年7月21日 · Netflix發起的Netflix Prize百萬美金競賽，是推薦系統領域最標誌性的事件之一
  - 比賽不但吸引了眾多專業人士開始投身於推薦系統領域的研究工作，也讓這項技術從學術圈真正地進入到了商業界
  - [Netflix Prize Dataset - Papers With Code](https://paperswithcode.com/dataset/netflix-prize)
    - https://www.kaggle.com/code/chahinebenali/content-based-and-tensorflow-recommender-system
    - https://www.kaggle.com/code/arunprakashkagg/recommendation-systems-2
    - https://www.kaggle.com/code/ianolmstead/netflix-recommendation-models
## 推薦引擎 
- Market basket analysis (association rule mining)
  - Ch2@BOOK2 Python推薦系統實戰：基於深度學習、NLP和圖算法的應用型推薦系統
  - https://github.com/Apress/applied-recommender-systems-python. 
- Content-based filtering(基於內容的推薦) ==> 文本特徵提取
- Collaborative-based filtering(協同過濾)
  - The two approaches of collaborative filtering
    - Memory
      - User-based and item-based collaborative filtering
    - Model-based 
- Hybrid systems
- ML clustering 
  - k-means clustering
  - fuzzy mapping
  - self-organizing maps (SOM)
  - a hybrid of two or more techniques
- ML classification
- Deep learning and NLP
  - Restricted Boltzmann
  - Autoencoder based
  - Neural attention–based
## IMDb 網路電影資料集 ==> 0 :負評   1:正評
- I can't remember many films where a bumbling idiot of a hero was so funny throughout. Leslie Cheung is such the antithesis of a hero that he's too dense to be seduced by a gorgeous vampire... I had the good luck to see it on a big screen, and to find a video to watch again and again. 9/10	 1 pos
- Master director Ching Siu Tung(程小東)'s perhaps most popular achievement is this series, A Chinese Ghost Story 1-3. Chinese Ghost Story stars Leslie Cheung in some distant past in China as a tax collector who is forced to spend a night during his "collecting trip" in a mysterious castle in which some strange old warriors fight and meet him. Beautiful actress Joey Wang/Wong is the ghost who lives in that castle and is under a domination of one powerful demon, a wood devil who collects human souls for herself/itself with the help of her beautiful ghosts. Leslie and Joey fall in love, and even though ghosts are not allowed to live with humans, they decide to break that rule and live happily together for the rest of their lives. This is not what the wood devil thinks and our protagonists have to fight for their lives and their happiness.<br /><br />This film is no less full of magic than other films by Ching Siu Tung. His masterpieces include Duel to the Death (1983) and the Swordsman series, which all have incredible visuals and kinetic power in their action scenes. Ghost Story is full of brilliant lightning and dark atmosphere, which is lightened by the strong presence of the beautiful and good willing ghost. The effects are simply breath taking and would work at their greatest power in the big screen. The camera is moving and twisted all the time and it adds to the fairy tale atmosphere this film has. There's plenty of wire'fu stunts, too, and even though some think they are and look gratuitous or stupid when used in films, I cannot agree and think they give motion pictures the kind of magic, freedom and creativeness any other tool could not give. When people fly in these films, it means the films are not just about our world, and they usually depict things larger than life with the power of this larger than life art form.<br /><br />The story about the power of love is pretty touching and warm, but the problem is (again) that the characters are little too shallow and act unexplainably occasionally. Leslie and Joey should have been written with greater care and their characters should be even more warm, deep and genuine in order to give the story a greater power and thus make the film even more noteworthy and important achievement. Also, the message about love and power of it is underlined little too much at one point and it should have been left just to the viewer's mind to be interpreted and found. Another negative point about the dialogue is that it's too plenty and people talk in this film without a reason. That is very irritating and sadly shows the flaws many scriptwriters tend to do when they write their movies. People just talk and talk and it's all there just to make everything as easy to understand as possible and so the film is not too challenging or believable as it has this gratuitous element. Just think about the films of the Japanese film maker Takeshi Kitano; his films have very little dialogue and all there is is all the necessary as he tells his things by other tools of cinema and never talks, or makes other characters talk too much in his movies. This is just the talent the writers should have in order to write greater scripts.<br /><br />Otherwise, Chinese Ghost Story is very beautiful and visually breath taking piece of Eastern cinema, and also the song that is played in the film is very beautiful and hopefully earned some award in the Hong Kong film awards back then. I give Chinese Ghost Story 7/10 and without the flaws mentioned above, this would without a doubt be almost perfect masterpiece of the fantasy genre.	 1pos

### 📊分析🎯⏱
- [EDA and Data Visualisation on the IMDb dataset](https://www.kaggle.com/code/rishabhbafnaiiitd/eda-and-data-visualisation-on-the-imdb-dataset)
- [GenAI Capstone 2025: MoodMatch Movie Recommender](https://www.kaggle.com/code/irisyang123/genai-capstone-2025-moodmatch-movie-recommender)
- [Content Based Recommendation System (IMDB dataset)](https://www.kaggle.com/code/sarthakrajimwale5/content-based-recommendation-system-imdb-dataset)
  - 兩種分析方法
    - 傳統方法（TF-IDF + 餘弦相似度）：將電影細節（如情節、類型和明星）轉換為向量，並測量它們之間的相似程度。
      - TF-IDF|Term Frequency-Inverse Document Frequency 
    - Deep 學習方法（神經嵌入）：使用深度學習來創建更豐富的電影表現形式，使我們能夠以更細緻的方式找到相似之處。 
