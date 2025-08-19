# from recommend_books import recommend_book
from logging import PlaceHolder
import numpy as np
import pandas as pd
import joblib
import streamlit as st

book_pivot = pd.read_csv("book_pivot.csv", index_col=0)
model = joblib.load("knn_model.pkl")


def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]

    distances, suggestions = model.kneighbors(
        book_pivot.iloc[book_id, :].values.reshape(1, -1)
    )

    # for i in range(len(suggestions)):
    #     if i != 0:
    #         print(f"The suggestions for {book_name}({book_id}) are :\n")
    #         print(book_pivot.index[suggestions[i]])
    #     else:
    #         print("None Found")

    # book_pivot.index[book_id]  # suggestions
    # # prints suggestions
    # for i in range(len(suggestions)):
    #     print(book_pivot.index[suggestions[i]])

    print(f"The suggestions for <<{book_name} ({book_id})>> are :\n")

    for idx in suggestions[0][1:]:  # skip index 0 (the book itself)
        print(book_pivot.index[idx])

    return suggestions


def main():
    st.set_page_config(
        page_title="Book Recommender",
        page_icon="📕📖📚📔",
        layout="centered",
        initial_sidebar_state="expanded",
        # initial_sidebar_state="collapsed",
    )

    st.title("Books Recommender System")

    user_book = st.text_input(
        label="Enter a book you've liked previously:",
        placeholder="Eg. - Harry Potter and the Chamber of Secrets",
    )

    submit = st.button("Generate recommendations")

    if submit:
        if user_book:
            result = recommend_book(user_book)
            st.text(
                f"The suggestions for <<{user_book} ({ np.where(book_pivot.index == user_book)[0][0]})>> are :\n"
            )
        for idx in result[0][1:]:  # skip index 0 (the book itself)
            # print(f"The suggestions for <<{book_name} ({book_id})>> are :\n")
            st.text(book_pivot.index[idx])

    # np.where(book_pivot.index == "Animal Farm")[0][0]

    recommend_book("Animal Farm")
    # recommend_book("Harry Potter and the Chamber of Secrets (Book 2)")


if __name__ == "__main__":
    main()
