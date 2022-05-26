---
layout: post
title: Variable length input for RNN models in PyTorch
tags: [ml]
comments: true
---

DataLoader: 
First define a Dataset class that hosts custom data. It needs to have __len__ and __getitem__ functions. In particular, __getitem__ returns the indexed element in the dataset.
DataLoader takes the custom dataset as input, and also a collate function is needed for the collate_fn argument. For variable length input sequences, we need to pad them so that all the sequences have the same length as the longest sequence. This can be done with pad_sequence. By default the padded locations are zeros. If the target label is also variable length (e.g. a seq2seq model), we also need to pad the targets with zeros. And during training, we need to define the ignore_index argument of the model to ignore zeros in the labels.


This is a demo post to show you how to write blog posts with markdown.  I strongly encourage you to [take 5 minutes to learn how to write in markdown](https://markdowntutorial.com/) - it'll teach you how to transform regular text into bold/italics/headings/tables/etc.

<!-- **Here is some bold text** -->

## Create a custom `Dataset` class dataset

Here's a useless table:

| Number | Next number | Previous number |
| :------ |:--- | :--- |
| Five | Six | Four |
| Ten | Eleven | Nine |
| Seven | Eight | Six |
| Two | Three | One |


How about a yummy crepe?

![Crepe](https://s3-media3.fl.yelpcdn.com/bphoto/cQ1Yoa75m2yUFFbY2xwuqw/348s.jpg)

It can also be centered!

![Crepe](https://s3-media3.fl.yelpcdn.com/bphoto/cQ1Yoa75m2yUFFbY2xwuqw/348s.jpg){: .mx-auto.d-block :}

Here's a code chunk:

~~~
var foo = function(x) {
  return(x + 5);
}
foo(3)
~~~

And here is the same code with syntax highlighting:

```javascript
var foo = function(x) {
  return(x + 5);
}
foo(3)
```

And here is the same code yet again but with line numbers:

{% highlight javascript linenos %}
var foo = function(x) {
  return(x + 5);
}
foo(3)
{% endhighlight %}

## Boxes
You can add notification, warning and error boxes like this:

### Notification

{: .box-note}
**Note:** This is a notification box.

### Warning

{: .box-warning}
**Warning:** This is a warning box.

### Error

{: .box-error}
**Error:** This is an error box.