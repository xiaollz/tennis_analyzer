"""EPUB stylesheet for bilingual FTT book."""

STYLESHEET = """\
body {
    font-family: Georgia, "Songti SC", "SimSun", serif;
    line-height: 1.7;
    margin: 1em 1.2em;
    color: #1a1a1a;
}

/* Chapter title */
h1 {
    font-size: 1.8em;
    font-weight: bold;
    text-align: center;
    margin-top: 2em;
    margin-bottom: 0.3em;
    page-break-before: always;
    color: #2c5f2d;
}
h1 .heading-zh {
    display: block;
    font-size: 0.6em;
    font-weight: normal;
    color: #666;
    margin-top: 0.2em;
}

/* Section heading */
h2 {
    font-size: 1.35em;
    font-weight: bold;
    margin-top: 1.5em;
    margin-bottom: 0.2em;
    color: #1a3a1a;
    border-bottom: 1px solid #ddd;
    padding-bottom: 0.15em;
}
h2 .heading-zh {
    display: block;
    font-size: 0.7em;
    font-weight: normal;
    color: #888;
}

/* Sub-section heading */
h3 {
    font-size: 1.12em;
    font-weight: bold;
    margin-top: 1.2em;
    margin-bottom: 0.15em;
    color: #2a4a2a;
}
h3 .heading-zh {
    display: block;
    font-size: 0.75em;
    font-weight: normal;
    color: #888;
}

h4 {
    font-size: 1.0em;
    font-weight: bold;
    margin-top: 1em;
    color: #3a5a3a;
}

/* English paragraph */
p.en {
    margin: 0.6em 0 0 0;
    text-align: justify;
}

/* Chinese translation */
p.zh {
    margin: 0.15em 0 0.6em 0;
    color: #555;
    font-size: 0.9em;
    text-align: justify;
    border-left: 3px solid #b8d4b8;
    padding-left: 0.7em;
}

/* Quote/callout box */
blockquote {
    margin: 1.2em 0.3em;
    padding: 0.7em 1em;
    background-color: #f0f7f0;
    border-left: 4px solid #2c5f2d;
}
blockquote p.en {
    font-style: italic;
    margin: 0.3em 0 0 0;
}
blockquote p.zh {
    border-left: none;
    padding-left: 0;
    font-style: normal;
    margin: 0.15em 0 0.3em 0;
}

/* Image + caption */
figure {
    text-align: center;
    margin: 1.5em 0;
    page-break-inside: avoid;
}
figure img {
    max-width: 100%;
    height: auto;
}
figcaption {
    font-size: 0.82em;
    color: #666;
    font-style: italic;
    margin-top: 0.3em;
    text-align: center;
}
figcaption .zh {
    display: block;
    font-style: normal;
    color: #888;
    margin-top: 0.1em;
}

/* Emphasis */
strong { font-weight: bold; }
em { font-style: italic; }

/* Cover page */
.cover-title {
    font-size: 2.2em;
    font-weight: bold;
    text-align: center;
    margin-top: 3em;
    color: #2c5f2d;
}
.cover-subtitle {
    font-size: 1.2em;
    font-style: italic;
    text-align: center;
    margin-top: 0.5em;
    color: #555;
}
.cover-author {
    font-size: 1.1em;
    text-align: center;
    margin-top: 1.5em;
}
.cover-edition {
    font-size: 0.85em;
    text-align: center;
    margin-top: 3em;
    color: #888;
}

/* Footnotes */
.footnote {
    font-size: 0.8em;
    color: #777;
    border-top: 1px solid #ddd;
    margin-top: 1.5em;
    padding-top: 0.5em;
}
"""
