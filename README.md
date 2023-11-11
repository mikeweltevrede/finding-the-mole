# Finding the Mole
[![build](https://github.com/mikeweltevrede/finding-the-mole/actions/workflows/ci.yml/badge.svg)](https://github.com/mikeweltevrede/finding-the-mole/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mikeweltevrede/finding-the-mole/graph/badge.svg?token=9VU08WT5PP)](https://codecov.io/gh/mikeweltevrede/finding-the-mole)
[![pre-commit](https://results.pre-commit.ci/badge/github/mikeweltevrede/finding-the-mole/main.svg)](https://results.pre-commit.ci/latest/github/mikeweltevrede/finding-the-mole/main)

Wie Is De Mol ("Who is the Mole") is a Dutch television show that has seen consistent popularity. It is also being
released in Belgium and there is an American version on Netflix. In this repository, I explore methods to predict who
the Mole will be.

## Contribution
I would love to receive your contributions to this learning project. You can contribute in multiple ways:
1. Help me with my [learning goals](#learning-goals) by sharing what you know on the topic or if you also have some
   questions you want answers to. Below, I explain what I  already know so please read that first. You can share by
   opening an issue.
2. Raise pull requests. Make sure to follow community standards with good descriptions, tests, and follow the
   [branch naming conventions](#branch-naming-conventions).

### Setup
1. Install [Poetry](https://python-poetry.org/) locally
   ```
   pip install poetry
   ```

2. Install the dependencies:
   ```
   poetry install
   ```

3. Activate pre-commit hooks:
   ```
   pre-commit install
   ```

4. Run the tests to see if the setup process was successful:
   ```
   poetry run pytest
   ```

### Branch naming conventions
A git branch should follow the following convention: `<category>/<description>/dev`.

A git branch should start with a category. Pick one of these: `feature`, `fix`, `hotfix`, or `test`.
- `feature` is for adding, refactoring, or removing a feature.
- `fix` is for fixing a bug.
- `hotfix` is for changing code with a temporary solution and/or without following the usual process (usually because of
  an emergency).
- `test` is for experimenting outside an issue/ticket.

The description should of course be descriptive; make sure that it reflects what you will be changing.

As such, correct examples are:
- `feature/add-ci-pipeline/dev`
- `fix/failing-unit-tests-for-new-polars-version/dev`
- `test/trying-out-polars/dev`

Inspiration: [A Simplified Convention for Naming Branches and Commits in Git](https://dev.to/varbsan/a-simplified-convention-for-naming-branches-and-commits-in-git-il4)

## Learning goals
I have set the following learning goals for myself so far:
1. I want to get familiar with using [Poetry](https://python-poetry.org/) for Python projects.
2. I want to get familiar with the [polars](https://www.pola.rs/) Python library.
3. I want to get familiar with the [deptry](https://github.com/fpgmaas/deptry) Python library.
4. I want to get familiar with [GitHub Actions](https://github.com/features/actions) to set up a CICD process.
5. I want to get familiar with using Bayesian modelling in Python (inspiration:
   [CodeErik](https://www.codeerik.nl/widm-2023-op-zoek-naar-de-mol-met-data-analyse/))

I will document my progress on each of these topics, and possibly more topics. I do this because I feel it is important
to share my learning path for others who might be looking into the same topics. Moreover, it is easy to forget where you
started once your efforts grow. Sometimes, it is good to look back at the basics and revise instead of only building on
existing resources.

If you want to help me along this journey, feel free to reach out! :)

### Poetry library
[**Poetry website**](https://python-poetry.org/)

I already knew about `poetry` but my interest recently spiked due to two events. Firstly, my team started to hire
new colleagues and many of them used `poetry`. If many of them did, surely it was worth looking into. Secondly, an
indirect colleague of mine joined the company and started to build upon the existing CICD templates. Previously, these
templates relied on a structure with `requirements.txt` files and did not yet support `poetry` projects. His dedication
showed me that there was definitely merit in the library.

I started my learning journey by watching YouTube videos. I noticed that many were relatively old, which makes sense
since `poetry` has been around since February 2018. The one I mainly used is
[How to Create and Use Virtual Environments in Python With Poetry](https://www.youtube.com/watch?v=0f3moPe_bhk)
by [ArjanCodes](https://www.arjancodes.com).

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

In the end, I saw in the documentation that it was possible to specify
[exact requirements](https://python-poetry.org/docs/dependency-specification/#exact-requirements) in the
`pyproject.toml` file.  This had my preference, so I removed the caret notation and specified specific versions. The
only annoying thing about this is that you have to either:
1. Look up the version you want to use and run `poetry add library@version`, or
2. Run `poetry add library`, remove the caret, delete the `poetry.lock` file, and rerun `poetry install` to create a new
   lock file.

To be fair, this problem also arises with regular `requirements.txt` files, so it is not an issue for me.

### Polars library
[**Polars website**](https://www.pola.rs/)

Polars grabbed my interest when I saw that it has a very similar syntax to [Pandas](https://pandas.pydata.org/), a
popular library for data analysis and data science, while being so much faster. Although for big data I like using
[Pyspark](https://spark.apache.org/docs/latest/api/python/index.html), I knew that that would be overkill for this
project. As such, I immediately knew that I wanted to try out Polars.

### Deptry library
[**Deptry GitHub repo**](https://github.com/fpgmaas/deptry)

I was first alerted to Deptry when a colleague of mine, the creator of the library, gave a presentation on his package
at work. I am keen on keeping my repo clean, so this solution to maintain the proper dependencies and clean them up
seemed perfect. Although it may not be as relevant or impactful for such a small project, it seems like a very nice
tool for larger projects.

### GitHub Actions
[**GitHub Actions website**](https://github.com/features/actions)

Before I started this project, I only heard of GitHub Actions through some GitHub templates. Since I wanted to make sure
to include CI into this repo, it seemed good to GitHub Actions for this purpose. I already had familiarity with YAML
pipelines through Azure DevOps so the learning curve is not that steep.

What I like about it so far is that you can easily take actions developed by other people into your pipeline. Moreover,
the triggers for the different workflows can be quite easily configured and also have many options. On the last point,
I did run into some issues when I wanted to adapt a workflow with a trigger "create" (run when a new branch is created);
namely, I could not manually (re)run that workflow on a new commit. It would be perfect if you can kick off any workflow
manually to test it. Perhaps you can and I could not find how. In that case, it is apparently not clear enough how you
could.

Overall, I do like how you can easily set up different files that are automatically configured to run on certain
triggers in your repo rather than having to activate the pipeline in a separate section.

### Bayesian modelling in Python
**Inspiration: [CodeErik](https://www.codeerik.nl/widm-2023-op-zoek-naar-de-mol-met-data-analyse/)**

### Other learnings from this project
#### Pre-commit hooks with public repo access
TODO
