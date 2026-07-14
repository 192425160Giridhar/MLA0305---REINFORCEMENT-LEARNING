###############################################################
# MARKOV DECISION PROCESS (MDP)
# COMPLETE R PROGRAM (MODULE 1 TO MODULE 6)
###############################################################

# Install packages (Run only once)
# install.packages(c("igraph","DiagrammeR","knitr"))

# Load Packages
library(igraph)
library(DiagrammeR)
library(knitr)

###############################################################
# MODULE 1
# STATES, ACTIONS, TRANSITIONS, PROBABILITIES AND REWARDS
###############################################################

cat("\n=====================================================\n")
cat("MODULE 1 : MARKOV DECISION PROCESS\n")
cat("=====================================================\n")

# States
states <- c("S1","S2","S3")

# Actions
actions <- c("A1","A2")

cat("\nNumber of States :", length(states), "\n")
cat("Number of Actions :", length(actions), "\n")

# Transition and Reward Data
mdp <- data.frame(
  
  CurrentState = c("S1","S1","S1","S1",
                   "S2","S2","S2","S2",
                   "S3","S3","S3","S3"),
  
  Action = c("A1","A2","A1","A2",
             "A1","A2","A1","A2",
             "A1","A2","A1","A2"),
  
  NextState = c("S2","S2","S3","S3",
                "S1","S1","S3","S3",
                "S1","S1","S2","S2"),
  
  Probability = c(0.6,0.4,0.2,0.8,
                  0.7,0.3,0.5,0.5,
                  0.9,0.1,0.4,0.6),
  
  Reward = c(5,10,-1,-5,
             3,7,2,1,
             4,6,0,-2)
)

cat("Number of Transitions :", nrow(mdp), "\n\n")

print(mdp)

###############################################################
# MODULE 2
# TRANSITION PROBABILITY MATRICES
###############################################################

cat("\n=====================================================\n")
cat("MODULE 2 : TRANSITION PROBABILITY MATRICES\n")
cat("=====================================================\n")

# Action A1 Matrix

A1 <- matrix(0,
             nrow=3,
             ncol=3,
             dimnames=list(states,states))

A1["S1","S2"] <- 0.6
A1["S1","S3"] <- 0.2

A1["S2","S1"] <- 0.7
A1["S2","S3"] <- 0.5

A1["S3","S1"] <- 0.9
A1["S3","S2"] <- 0.4

cat("\nTransition Probability Matrix (Action A1)\n")
print(A1)


# Action A2 Matrix

A2 <- matrix(0,
             nrow=3,
             ncol=3,
             dimnames=list(states,states))

A2["S1","S2"] <- 0.4
A2["S1","S3"] <- 0.8

A2["S2","S1"] <- 0.3
A2["S2","S3"] <- 0.5

A2["S3","S1"] <- 0.1
A2["S3","S2"] <- 0.6

cat("\nTransition Probability Matrix (Action A2)\n")
print(A2)

###############################################################
# MODULE 3
# REWARD MATRIX
###############################################################

cat("\n=====================================================\n")
cat("MODULE 3 : REWARD TABLE\n")
cat("=====================================================\n")

reward_table <- mdp[,c("CurrentState",
                       "Action",
                       "NextState",
                       "Reward")]

kable(reward_table)

###############################################################
# MODULE 4
# EXPECTED IMMEDIATE REWARD
###############################################################

cat("\n=====================================================\n")
cat("MODULE 4 : EXPECTED IMMEDIATE REWARD\n")
cat("=====================================================\n")

summary_table <- data.frame()

for(s in states){
  
  cat("\n-------------------------------------\n")
  cat("Current State :", s, "\n")
  
  for(a in actions){
    
    temp <- subset(mdp,
                   CurrentState==s &
                     Action==a)
    
    expected_reward <- sum(temp$Probability *
                             temp$Reward)
    
    cat("\nAction :", a,"\n")
    
    print(temp)
    
    cat("\nCalculation\n")
    
    for(i in 1:nrow(temp)){
      
      cat(temp$Probability[i]," * ",
          temp$Reward[i],"\n")
    }
    
    cat("Expected Immediate Reward =",
        round(expected_reward,2),"\n")
    
    summary_table <- rbind(summary_table,
                           
                           data.frame(
                             State=s,
                             Action=a,
                             ExpectedReward=
                               round(expected_reward,2)
                           ))
  }
}

###############################################################
# MODULE 5
# SUMMARY TABLE
###############################################################

cat("\n=====================================================\n")
cat("MODULE 5 : OUTPUT SUMMARY\n")
cat("=====================================================\n")

kable(summary_table)

###############################################################
# MODULE 6
# VISUALIZATION
###############################################################

cat("\n=====================================================\n")
cat("MODULE 6 : STATE TRANSITION DIAGRAM\n")
cat("=====================================================\n")

edges <- data.frame(
  
  from = mdp$CurrentState,
  
  to = mdp$NextState,
  
  label = paste(mdp$Action,
                "\nP=",
                mdp$Probability)
  
)

g <- graph_from_data_frame(edges,
                           directed=TRUE)

plot(
  g,
  
  vertex.size=45,
  
  vertex.color=c("gold",
                 "skyblue",
                 "lightgreen"),
  
  vertex.frame.color="black",
  
  vertex.label.cex=1.5,
  
  vertex.label.color="black",
  
  edge.label=edges$label,
  
  edge.label.cex=0.8,
  
  edge.label.color="blue",
  
  edge.arrow.size=0.5,
  
  main="Markov Decision Process"
)

###############################################################
# COLORFUL DIAGRAM USING DIAGRAMMER
###############################################################

grViz("

digraph MDP {

graph [layout = dot, rankdir = LR]

node [shape=circle,
style=filled,
fontsize=18,
fontcolor=black]

S1 [fillcolor=gold]
S2 [fillcolor=skyblue]
S3 [fillcolor=lightgreen]

S1 -> S2 [label='A1 (0.6)', color='blue']
S1 -> S2 [label='A2 (0.4)', color='red']

S1 -> S3 [label='A1 (0.2)', color='blue']
S1 -> S3 [label='A2 (0.8)', color='red']

S2 -> S1 [label='A1 (0.7)', color='blue']
S2 -> S1 [label='A2 (0.3)', color='red']

S2 -> S3 [label='A1 (0.5)', color='blue']
S2 -> S3 [label='A2 (0.5)', color='red']

S3 -> S1 [label='A1 (0.9)', color='blue']
S3 -> S1 [label='A2 (0.1)', color='red']

S3 -> S2 [label='A1 (0.4)', color='blue']
S3 -> S2 [label='A2 (0.6)', color='red']

}

")

###############################################################
# END OF PROGRAM
###############################################################