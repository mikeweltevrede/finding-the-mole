# Finding the Mole
[![build](https://github.com/mikeweltevrede/finding-the-mole/actions/workflows/ci.yml/badge.svg)](https://github.com/mikeweltevrede/finding-the-mole/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mikeweltevrede/finding-the-mole/graph/badge.svg?token=9VU08WT5PP)](https://codecov.io/gh/mikeweltevrede/finding-the-mole)
[![pre-commit](https://results.pre-commit.ci/badge/github/mikeweltevrede/finding-the-mole/main.svg)](https://results.pre-commit.ci/latest/github/mikeweltevrede/finding-the-mole/main)

Wie Is De Mol ("Who is the Mole") is a Dutch television show that has seen consistent popularity. It is also being released in Belgium and there is an American version on Netflix. In this repository, I explore methods to predict who the Mole will be.

## Learning goals
I have set the following learning goals for myself so far:
1. I want to get familiar with using [Poetry](https://python-poetry.org/) for Python projects.
1. I want to get familiar with the [polars](https://www.pola.rs/) Python library.
1. I want to get familiar with the [deptry](https://github.com/fpgmaas/deptry) Python library.
1. I want to get familiar with GitHub Actions to set up a CICD process.
1. I want to get familiar with using Bayesian modelling in Python (inspiration: [CodeErik](https://www.codeerik.nl/widm-2023-op-zoek-naar-de-mol-met-data-analyse/))

I will document my progress on each of these topics, and possibly more topics. I do this because I feel it is important
to share my learning path for others who might be looking into the same topics. Moreover, it is easy to forget where you
started once your efforts grow. Sometimes, it is good to look back at the basics and revise instead of only building on
existing resources.

If you want to help me along this journey, feel free to reach out! :)

## Poetry
I already knew about `poetry` but my interest recently spiked due to two events. Firstly, my team started to hire
new colleagues and many of them used `poetry`. If many of them did, surely it was worth looking into. Secondly, an
indirect colleague of mine joined the company and started to build upon the existing CICD templates. Previously, these
templates relied on a structure with `requirements.txt` files and did not yet support `poetry` projects. His dedication
showed me that there was definitely merit in the library.

I started my learning journey by watching YouTube videos. I noticed that many were relatively old, which makes sense
since `poetry` has been around since February 2018. The one I mainly used is
[How to Create and Use Virtual Environments in Python With Poetry](https://www.youtube.com/watch?v=0f3moPe_bhk)
by [ArjanCodes](www.arjancodes.com).

Of course, I had to start with setting up my project with `poetry`. I noticed that it was easier to start locally and
create my project folder with `poetry new` and then running `git init`, rather than cloning an already initialized
GitHub repo from the website and calling `poetry init`.

Secondly, I was wondering about the `poetry.lock` file and whether I should commit it. The
[official website says](https://python-poetry.org/docs/basic-usage/#:~:text=You%20should%20commit%20the%20poetry.lock%20file%20to%20your%20project%20repo%20so%20that%20all%20people%20working%20on%20the%20project%20are%20locked%20to%20the%20same%20versions%20of%20dependencies):

> When Poetry has finished installing, it writes all the packages and their exact versions that it downloaded to the
> poetry.lock file, locking the project to those specific versions. You should commit the poetry.lock file to your
> project repo so that all people working on the project are locked to the same versions of dependencies.

However, when reviewing the applications that I alluded to earlier, I had some issues specifically pertaining to the
lock file. I also read [the following comment by Arne on StackOverflow](https://stackoverflow.com/a/61076546):
> My personal experience has been that it isn't necessary to commit lockfiles to VCS. The pyproject.toml file is the
> reference for correct build instructions, and the lockfile is the reference for a single successful deployment. Now, I
> don't know what the spec for poetry.lock is, but I had them backfire on me often enough during collaboration with
> colleagues in ways where only deleting them would fix the problem. [...] I help maintain a number of closed and open
> source projects, and they all commit lockfiles, partly because I advocated in favor of it. By now I regret that
> choice, because it occurs quite often that someone's build is not working and the solution is to delete and re-build
> the lockfile, after which all of us end up having merge conflicts.

To be continued...
