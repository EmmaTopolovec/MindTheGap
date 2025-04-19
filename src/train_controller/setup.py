from setuptools import setup

package_name = 'train_controller'

setup(
    name=package_name,
    version='1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Emma Topolovec',
    maintainer_email='etopolo1@binghamton.edu',
    description='Train and door control logic for Gazebo simulation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'train_control_node = train_controller.train_control_node:main',
        ],
    },
)
