# add stuff to command line
export PATH=${PATH}:/usr/local/mysql/bin


# basic prompt
# PS1='\u@\h \w \$ '
export PS1="________________________________________________________________________________\n| \w @ \h (\u) \n| => "
export PS2="| => "

# remove computer name from beginig on command line
# More about it:
# http://www.cyberciti.biz/tips/howto-linux-unix-bash-shell-setup-prompt.html
#export PS1="$ "

# color prompt
# - see https://wiki.archlinux.org/index.php/Color_Bash_Prompt
PS1="$(if [[ ${EUID} == 0 ]]; then echo '\[\033[01;31m\]\h'; else echo '\[\033[01;32m\]\u@\h'; fi)\[\033[01;34m\] \w \$([[ \$? != 0 ]] && echo \"\[\033[01;31m\]:(\[\033[01;34m\] \")\\$\[\033[00m\] "

# show "[hh:mm] user@host:pwd" in xterm title bar
if [ "$TERM_PROGRAM" = "Apple_Terminal" ]; then
    # for Mac Terminal, omit "User@Users-MacBook-Air"
    # and preserve PROMPT_COMMAND set by /etc/bashrc.
    show_what_in_title_bar='"[$(date +%H:%M)] ${PWD/#$HOME/~}"'
    PROMPT_COMMAND='printf "\033]0;%s\007" '"$show_what_in_title_bar; $PROMPT_COMMAND"
else
    show_what_in_title_bar='"[$(date +%H:%M)] ${USER}@${HOSTNAME%%.*}:${PWD/#$HOME/~}"'
    case ${TERM} in
        xterm*|rxvt*|Eterm|aterm|kterm|gnome*)
            PROMPT_COMMAND='printf "\033]0;%s\007" '$show_what_in_title_bar
            ;;
        screen)
            PROMPT_COMMAND='printf "\033_%s\033\\" '$show_what_in_title_bar
            ;;
    esac
fi
unset show_what_in_title_bar


# use vim as a default command line editor
export EDITOR="vim"

# set vi editing mode
set -o vi

# enable bash autocompletion
[ -f /usr/local/etc/bash_completion ] && . /usr/local/etc/bash_completion

# easier folders navigation
alias ..="cd .."
alias ...="cd ../.."
alias ....="cd ../../.."
alias .....="cd ../../../.."

# utility shortcuts improved
alias ls="ls -F" # show trailing slash with folders
alias ll="ls -laF" # show list of all files (including hidden)
alias cl='cd; ll'

# some git shortcuts
alias g="git"
alias gs="git status"
alias gl="git log --graph --oneline --all --decorate"
alias gc="git commit"
alias ga="git add -A"
alias gb="git branch"
alias gco="git checkout"
alias gacm="git add -A && git commit -m"
alias gprm="git pull --rebase origin master"
alias grep="grep --color=auto"
alias egrep="egrep --color=auto"
alias fgrep="fgrep --color=auto"

# why your mac is so slow?
alias top="top -o vsize"

# run a dead simple python server
alias serve="echo 'Your cool server is runing on http://localhost:8000/' && open http://localhost:8000/ && python -m SimpleHTTPServer 8000"
alias servephp="echo 'Your cool server is runing on http://localhost:8000/' && open http://localhost:8000/ && php -S 0.0.0.0:8000 -t ."

# start / stop apache & MySQL
alias am-start="sudo apachectl start && mysql.server start"
alias am-stop="sudo apachectl stop && mysql.server stop"

# npm shit
alias npmi="npx benny-hill npm i"
alias npmu="npx benny-hill npm u"
alias npmt="npx benny-hill npm t"

# sudo autocomplete
complete -cf sudo

svndiff()     { svn diff "$@" | colordiff; }
svndiffless() { svn diff "$@" | colordiff | less -R; }

alias gccg='gcc -g -Wall'
alias g++g='g++ -g -Wall'
alias valgrindlc='valgrind --leak-check=yes'


export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
