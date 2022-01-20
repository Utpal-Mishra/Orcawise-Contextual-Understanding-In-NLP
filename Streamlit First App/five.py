"""# Plotting the wordcloud
import matplotlib.pyplot as plt
 
 
from wordcloud import WordCloud, STOPWORDS
 
# Creating a custom list of stopwords
customStopwords=list(STOPWORDS) + ['less','Trump','American','politics','country']
 
wordcloudimage = WordCloud( max_words=50,
                            font_step=2 ,
                            max_font_size=500,
                            stopwords=customStopwords,
                            background_color='black',
                            width=1000,
                            height=720
                          ).generate(NewNounString)
 
plt.figure(figsize=(20,8))
plt.imshow(wordcloudimage)
plt.axis("off")
plt.show()"""