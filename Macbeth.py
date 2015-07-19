
# coding: utf-8

# # WordsCount_Text from Macbeth Shakespeare

# Lines count

# In[11]:

#Read text
lines = sc.textFile('Macbeth.txt')
lines.take(10)


# In[12]:

#Count Lines
lines.count()


# In[13]:

#Count lines after remove empty lines
lines_nonempty = lines.filter( lambda x: len(x) > 0 )
lines_nonempty.count()


# Words count

# In[4]:

lines_nonempty.take(10)


# In[14]:

#Replace symbols to space
a1=lines.map(lambda x: x.replace(',',' ').replace('.',' ').replace('-',' ').replace('?',' ').replace('[',' ').replace(']',' ').lower())
a1.take(10)


# In[15]:

#Devide lines to words
a2 = a1.flatMap(lambda x: x.split())
a2.take(10)


# In[16]:

#Create tuple which has unique words
a3 = a2.map(lambda x: (x, 1))
a3.take(10)


# In[8]:

#Sum of each words
a4 = a3.reduceByKey(lambda x,y: x+y)
a3.take(10)


# In[9]:

#Swap the order
a5 = a4.map(lambda x:(x[1],x[0]))
a5.take(10)


# In[10]:

#Sort by the number of words
a6 = a5.sortByKey(ascending=False)
a6.take(10)


# In[ ]:



