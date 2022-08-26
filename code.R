###############################################################################
# 词嵌入方法实例：在新闻联播中找出疫情相关报道
# 作者：陈昊瀚
# 注：本实例所用数据由王弈强（香港大学）、陈昊瀚（香港大学）从央视网收集
# 整理。本代码由陈昊瀚编写。欢迎各位同业先进将本数据和代码用于学习、科研、教学。
# 本实例所用工作流程的详细描述和其他应用参见：
#   https://haohanchen.github.io/files/ChenHaohan_word_embedding.pdf
#   https://link.springer.com/article/10.1007/s11266-021-00399-7
###############################################################################

rm(list=ls())

# 加载包 ----

## 数据管理包
library(tidyverse)
library(lubridate)

## 分词工具
library(jiebaR)

## 词嵌入工具
library(text2vec)

## Excel文件读写工具（用于中间的手工标记数据步骤）
library(openxlsx)


# 加载数据 ----

d <- readRDS("data/data_xwlb.rds")


# 分词 ----

# 分词
tokenizer <- worker()
d$text_tok <- lapply(d$text, function(x) segment(x, tokenizer))

# 看看分词后每篇文档大概多少个词
hist(sapply(d$text_tok, length), 
     xlab = "词数", ylab = "频率",
     main = "每篇报道词数", family = "Kai")


# 寻找分词器未发现的固定词组 ----

## 参阅: https://text2vec.org/collocations.html

model_collocation <- Collocations$new(collocation_count_min = 10)
## collocation_count_min = 10 固定搭配需最少出现10次才能入选
## 这个数值越高，入选的词组越少

it <- itoken(d$text_tok)
## 把分词好的文本放入一个循环容器（以便训练模型时高效读取）

model_collocation$fit(it, n_iter = 3)
## n_iter = 3 进行3轮常见搭配搜寻。每一轮过后把找到的搭配作为一个词放进下一轮
## 这个数值设得越大，找到的最长词组越长（根据自己需求，长词组多不一定好）

result_collocation <- model_collocation$collocation_stat %>%
  as_tibble() %>%
  mutate(Selected = NA) %>%
  arrange(desc(n_ij)) %>% # 按固定搭配出现的频次排序。优先考虑高频词组
  select(Selected, everything())

write.xlsx(result_collocation, "data/output/collocation_blank.xlsx", overwrite = FALSE)
write.xlsx(result_collocation, "data/output/collocation_label.xlsx", overwrite = FALSE)

# 手工寻找有意义的固定搭配 ----

## 打开导出的excel文件collocation_label.xlsx 手工标记有意义的固定搭配
## 什么样的词组是有意义的，要看研究者的需要。没有对错之分。


# 运用手工挑出来的词组改善分词质量 ----

rm(result_collocation)

## 导入手工标记好的*有用*固定搭配 ----

result_collocation <- read.xlsx("data/output/collocation_label.xlsx") %>%
  filter(Selected == 1) %>%
  mutate(phrase = paste0(prefix, suffix)) %>% # 把固定搭配的两部分“粘”在一起
  mutate(phrase = str_remove_all(phrase, "_")) # 去掉中间的分隔符 "_"
  
result_collocation$phrase[1:10]

## 初始化一个新的分词器 ----
tokenizer <- worker()
segment("新冠肺炎", tokenizer) # 随机检查：此时分词器并不认识“新冠肺炎”这个词组

## 将标记好的新固定搭配输入分词器 ---
new_user_word(tokenizer, result_collocation$phrase)
segment("新冠肺炎", tokenizer) # 随机检查：现在分词器认识“新冠肺炎”了！

## 用“加强版”分词器重做一遍分词 ----
d$text_tok <- lapply(d$text, function(x) segment(x, tokenizer))

# Some simple diagnostics 看看分词后每篇文档大概多少个词
hist(sapply(d$text_tok, length), 
     xlab = "词数", ylab = "频率",
     main = "每篇报道词数", family = "Kai")


# 训练词嵌入模型 ---- 

## 准备工作 ----
it <- itoken(d$text_tok) # 把分词好的文本导入text2vec的数据容器
vocab <- create_vocabulary(it) # 创造词典. 若数据大，这一步耗时长

summary(vocab) # 看看词典长什么样
### term: 词
### term_count: 词语总共出现了几次
### doc_count: 词语在几个文档里出现过

vocab <- prune_vocabulary(vocab, term_count_min = 20)
# term_count_min = 20 词语至少出现20次才能入选

vectorizer <- vocab_vectorizer(vocab)
tcm <- create_tcm(it, vectorizer, skip_grams_window = 10)

## 训练模型 ----

model_glove = GlobalVectors$new(rank = 50, x_max = 10)
wv_main <- model_glove$fit_transform(tcm, n_iter = 10, convergence_tol = 0.01, n_threads = 8)
dim(wv_main)
wv_context <- model_glove$components
dim(wv_context)

## 取得词向量 ----

word_vectors <- wv_main + t(wv_context)

saveRDS(model_glove, "data/output/embedding_model.rds") # 储存模型，可重复利用
saveRDS(word_vectors, "data/output/embedding_wordvector.rds") # 储存词向量


# 初步评估训练好的词嵌入模型 ----

# Helper function: Average multiple word vectors
get_wv_average <- function(query_terms){
  query_terms_s <- query_terms[query_terms %in% vocab$term]
  query_wv <- matrix(colMeans(word_vectors[query_terms_s, , drop = FALSE]), nrow = 1)
  query_wv
}

# Helper function: Calculate cosine similarities
get_cossim <- function(matrix1, matrix2){
  cos_sim <- sim2(x = matrix1, y = matrix2, method = "cosine", norm = "l2")
  cos_sim
}

# Helper function: Get top similar words
get_top_similar_words <- function(query_terms, top_n = 20){
  query_wv <- get_wv_average(query_terms)
  cos_sim <- get_cossim(word_vectors, query_wv)
  head(sort(cos_sim[,1], decreasing = TRUE), top_n) %>%
    as.data.frame() %>%
    rownames_to_column() %>%
    as_tibble() %>%
    rename("cosine_similarity" = ".", "term_similar" = "rowname") %>%
    mutate(term_query = paste(query_terms, collapse = "-")) %>%
    select(term_query, term_similar, cosine_similarity) %>%
    inner_join(vocab, by = c("term_similar" = "term"))
  
}

get_top_similar_words("上海", 20)
get_top_similar_words("抗疫", 20)
get_top_similar_words("扶贫", 20)
get_top_similar_words("发展", 20)
get_top_similar_words(c("美国", "俄罗斯"), 20)


# 运用词向量来扩大词典 ----

dictionary_expand <- get_top_similar_words(c("疫情", "抗疫"), 200)

dictionary_expand <- dictionary_expand %>% 
  mutate(Select = NA) %>% select(Select, everything())

write.xlsx(dictionary_expand, "data/output/dictionary_expand_blank.xlsx", overwrite = TRUE)
write.xlsx(dictionary_expand, "data/output/dictionary_expand_label.xlsx", overwrite = FALSE)

rm(dictionary_expand)


# 手工标记扩大的词典 ----

## 打开excel文件dictionary_expand_label.xlsx
## 根据研究者需要，决定哪些词是正确的“扩展”


# 加载手工标记的扩大词典 ----

out_query_vocab <- read.xlsx("data/output/dictionary_expand_label.xlsx") %>%
  filter(Select == 1)

# 构造“概念”向量：求词典的词向量平均值 ----
out_query_vector <- get_wv_average(out_query_vocab$term_similar)


# 构造文档向量 ----

document_vectors <- t(sapply(d$text_tok, function(x) get_wv_average(x)))
## 此行代码较花时间
## 此处我用了最简单的count方法。可尝试TF-IDF权重算平均值，减少常见词/停用词的影响。


# 构建测量值 ----

## 我们感兴趣的“测量”是每篇文档跟我们关心的词典之间的相关度
## 有两种方法可构建这个测量值

## 词典“数豆子”法 (不连续变量，精确，但可能不齐全) ----

d$relevance_count <- sapply(
  d$text_tok, 
  function(x) 
  length(intersect(out_query_vocab$term_similar, x)))

hist(d$relevance_count, 
     main = "关键词频数分布", xlab = "频数", ylab = "文档数",
     family = "Kai")


## 词向量与文档向量余弦相似性法 （连续变量，可能不精确） ----

d$relevance_cossim <- get_cossim(document_vectors, out_query_vector)[,1]

hist(d$relevance_cossim, 
     main = "余弦相似度分布", xlab = "频数", ylab = "文档数",
     family = "Kai")


# 保存结果 ----

d_out <- d %>%
  select(date, order, text, relevance_count, relevance_cossim)

write.xlsx(d_out, "data/output/labeled_documents.xlsx")



# 随便看看数据趋势 ----

## 看看疫情报道的数量随时间的变化

sum_stats <- d_out %>%
  group_by(date) %>%
  summarise_at(vars(relevance_count, relevance_cossim), mean)

ggplot(sum_stats, aes(x = date, y = relevance_count)) +
  geom_line() +
  geom_smooth() +
  labs(
    title = "2020上半年新闻联播有关疫情报道的含量",
    subtitle = "(关键词平均含量)",
    caption = "制图: 陈昊瀚") +
  xlab("日期") + ylab("关键词平均含量") +
  theme(title = element_text(family = "Kai"))

ggplot(sum_stats, aes(x = date, y = relevance_cossim)) +
  geom_line() +
  geom_smooth() +
  labs(
    title = "2020上半年新闻联播有关疫情报道的含量",
    subtitle = "(与关键词组向量余弦相似度)",
    caption = "制图: 陈昊瀚") +
  xlab("日期") + ylab("平均余弦相似度") +
  theme(title = element_text(family = "Kai"))

### 注：执行以上两行代码出错，大概率是中文字体的问题。可先移除theme()一行
### 移除这一行后应该图可画出，但是标题和坐标轴标题会现实成方框
### 这个问题应该windows系统上更易出现
