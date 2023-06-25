# Imports
import tkinter as tk
import textwrap
from Common import setWindowStyle
from Backend.GetEmbedding import creatEmbedding, twoEmbeddings, cosine_similarity_percentage


##########################################################################################################################
class displayResults:
    def __init__(self, selected_item):
        self.root = tk.Tk()
        setWindowStyle(self.root,
                       "PART : Pre-trained Authorship Embedding Transformer>choosing document>Document’s authorship embedding display",
                       "900x700")
        self.embedding_dataset = creatEmbedding()
        self.shakespeareEm, self.imposterEm = twoEmbeddings(self.embedding_dataset, selected_item)

        label1 = tk.Label(self.root, text="Authorship Embeddings:", font=('Ariel', 38), fg="whitesmoke", bg="#282828")
        label1.grid(column=0, row=0, sticky="NSEW")

        labelsFrame = tk.Frame(self.root)
        labelsFrame.configure(bg="#282828")
        labelsFrame.grid(column=0, row=1)

        label2 = tk.Label(labelsFrame, text="The Embedding for\nWilliam Shakespeare", font=('Ariel', 25, "underline"),
                          fg="whitesmoke",
                          bg="#282828")
        label2.grid(column=0, row=0, sticky="N")
        self.selected_text = selected_item
        stringEm = []
        for num in self.shakespeareEm[:30]:
            stringEm.append(str(num))
        string = "{ " + ", ".join(stringEm) + ",...}"
        # Wrap this text.
        wrapper = textwrap.TextWrapper()
        word_list = wrapper.wrap(text=string)

        for i, element in enumerate(word_list):
            label = tk.Label(labelsFrame, text=element, font=('Arial', 10), fg="whitesmoke", bg="#282828")
            label.grid(column=0, row=1 + i)

        #######################################################################################################
        imposter_book = ""
        word_list_book = selected_item.split(" ")
        i = 1
        for word in word_list_book:
            imposter_book += (word + " ")
            if i % 4 == 0:
                imposter_book += ("\n")
            i = i + 1

        label3 = tk.Label(labelsFrame, text="The embedding for\nthe writer of\n\"" + imposter_book + "\"",
                          font=('Ariel', 25, "underline"),
                          fg="whitesmoke",
                          bg="#282828")
        label3.grid(column=1, row=0)

        # Show THE Imposter Embedding
        stringEm = []
        for num in self.imposterEm[:30]:
            stringEm.append(str(num))
        string = "{ " + ", ".join(stringEm) + ",...}"
        word_list = wrapper.wrap(text=string)
        for j, element in enumerate(word_list):
            label = tk.Label(labelsFrame, text=element, font=('Arial', 10), fg="whitesmoke", bg="#282828")
            label.grid(column=1, row=1 + j)

        resultFrame = tk.Frame(self.root, height=20, bd=0)
        resultFrame.configure(bg="#282828", relief="solid", highlightbackground="#00bcd4", highlightthickness=1)
        resultFrame.grid(column=0, row=2 + max(i, j), )

        label4 = tk.Label(resultFrame, text="Comparing:", font=('Ariel', 20), fg="whitesmoke", bg="#282828")
        label4.pack()

        label5 = tk.Label(resultFrame, text="The embeddings are", font=('Ariel', 20), fg="whitesmoke", bg="#282828")
        label5.pack()

        percentage = tk.Label(resultFrame, font=('Ariel', 20, "bold"), fg="whitesmoke",
                              bg="#282828")
        # Calculate the cosine similarity
        cosine_sim = cosine_similarity_percentage(self.shakespeareEm, self.imposterEm)
        rounded_distance = round(cosine_sim, 2)
        percentageTxt = str(rounded_distance) + "%"
        percentage.config(text=percentageTxt)
        percentage.pack()

        label6 = tk.Label(resultFrame, text="alike.", font=('Ariel', 20), fg="whitesmoke", bg="#282828")
        label6.pack()

        num_rows = self.root.grid_size()[1]
        index = 0
        for row in range(num_rows):
            if index == 1:
                self.root.grid_rowconfigure(row, weight=2)
            else:
                self.root.grid_rowconfigure(row, weight=1)
        self.root.grid_rowconfigure(num_rows - 1, weight=2)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.mainloop()
